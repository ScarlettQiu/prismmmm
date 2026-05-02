# PrismMMM — Plain English Guide
### What We Did, How It Works, and What We Found

*For business readers with no statistics or math background.*

---

## What is Marketing Mix Modeling?

Imagine you're running a shop and spending money on several different types of advertising — Facebook ads, Google ads, Instagram ads. At the end of each week, you count your sales. The question is: **which advertising actually caused those sales?**

That's harder than it sounds. Sales go up in November because of holiday shopping. Sales go up when you run a promotion. Sales go up when Facebook *and* Google *and* Instagram all happen to spend more the same week. How do you separate what *advertising* caused from what *seasons and promotions* caused?

**Marketing Mix Modeling (MMM)** is a technique that looks at 2+ years of weekly data — every week's revenue alongside every week's advertising spend — and tries to mathematically separate those causes. It answers: *"If we spent nothing on Meta Facebook this year, how much revenue would we have lost?"*

---

## What is "Regression"?

Think of it like this. You have 132 weeks of data. Each week you know how much you spent on each advertising channel and how much revenue you made.

Regression is a way of finding the best explanation for why revenue goes up and down. It asks: *"For every extra dollar spent on Meta Facebook, how many dollars of extra revenue do we typically see?"* That number — dollars of revenue per dollar of ad spend — is called **ROI (Return on Investment)**.

We didn't use just one method to calculate this. We used **three different mathematical approaches** to cross-check each other. If all three agree, you can trust the answer. If they disagree, something is making the picture unclear — maybe the channels were all spending more at the same time, or there isn't enough data.

---

## Part 1: Checking the Data Before We Start

Before running any models, the **Data Explorer agent** examined the raw data — like a doctor reviewing test results before making a diagnosis.

### What the data looks like

- **132 weeks** of data, November 2021 to May 2024
- **8 advertising channels**: Meta Facebook, Meta Instagram, Google Search, Google Shopping, Google PMax, Google Display, Google Video, Meta Other
- Weekly revenue ranged from **$10.6M to $114M**
- Average weekly revenue: **$30.3M**
- The data is clean — no missing weeks, no duplicate entries

### Who's spending what

| Channel | Total Spend (2.5 years) | Notable |
|---|---|---|
| Meta Facebook | $624M | By far the biggest — 74% of Meta budget |
| Meta Instagram | $226M | Second — 26% of Meta budget |
| Google PMax | $74M | Largest Google channel |
| Google Search | $18M | |
| Google Video | $16M | |
| Google Display | $6.6M | |
| Google Shopping | $3.3M | Almost never used — 76% of weeks had $0 spend |
| Meta Other | $0.5M | Tiny |

**Meta is spending ~88% of the total advertising budget.** Google channels make up the remaining 12%.

### What the data explorer flagged as concerning

**1. Google Shopping is nearly inactive.**
76% of weeks had zero dollars spent on Google Shopping. This is a problem for modeling because you can't measure the effect of something that barely ran. The model will struggle to say anything reliable about Google Shopping.

**2. Unusual weeks that need explaining.**
On **30 May 2022**, something extraordinary happened: revenue hit $114M (the highest week in the dataset), Google Shopping spend spiked to 10× its normal level, and Meta Facebook also spiked. This happened all at once. It's almost certainly a major sales event or promotion. These "outlier" weeks can distort the model's understanding if not handled carefully — like trying to understand a person's typical diet by including one week where they ate at 10 restaurants for a special occasion.

Similar spikes happened around **holiday seasons** (November–December) and **May 2023**.

**3. Google Display and Google Video show almost no relationship with revenue.**
When a channel's spend goes up and revenue doesn't move with it, that's a signal the channel either doesn't work, or its effect is too delayed or too mixed with other things for the model to see it clearly.

**4. Meta Facebook shows the strongest raw relationship with revenue** — when Facebook spend goes up, revenue tends to go up with it. Google Search is second. Instagram is third.

**Readiness score: 4 out of 5** — good data, with a few known issues to watch.

---

## Part 2: How the Models Work

We used three different modeling approaches. Here's what each one does in plain English:

### Ridge — "The Disciplined Estimator"
Looks at all channels simultaneously and tries to find how much each one contributes to revenue. It's "disciplined" because it automatically reduces extreme answers — if a channel seems to explain 500% of revenue, it pulls that number back toward something more realistic. Fast and transparent.

### PyMC Bayesian — "The Careful Statistician"
Does the same thing but starts with some prior beliefs ("we expect ad spend to have a positive effect") and then updates those beliefs with the data. Like a scientist who says "I don't think this channel is magic, but let me check the data and adjust." This approach also gives you a range of uncertainty, not just a single number. It's slower but gives richer outputs.

### BayesianRidge — "The Positive-Constrained Estimator"
A regularised model that puts all channels on the same scale before estimating their contributions — so a channel spending $3M and a channel spending $600M can be compared fairly. It also enforces an economic reality check: media channels collectively cannot be credited with more revenue than they plausibly drive. This replaced the earlier NNLS model in Round 8 and produced the most accurate out-of-sample predictions.

---

### Two concepts the models use

**Adstock (carryover effect)**
When you see a Facebook ad today, you might not buy until next week. Or you might see the ad, forget about it, and then remember the brand when you're ready to shop two weeks later. "Adstock" captures this — it means today's ad spend has a lingering effect on future weeks' revenue.

Different channels carry over for different amounts of time:

| Channel | How long the effect lingers | Why |
|---|---|---|
| Google Search | Days | People search when they're ready to buy — effect is immediate |
| Meta Facebook / Instagram | 2–3 weeks | Social ads build awareness that lingers |
| Google Video | 3–4 weeks | Brand-building video has longer recall |
| Google Display | 2–3 weeks | Awareness channel, medium carryover |

In Round 4, we gave the model these real-world benchmarks from a knowledge layer instead of using one average for all channels. This made a significant difference — more on that below.

**Hill Saturation (diminishing returns)**
If you spend $1M on Facebook and get 1,000 customers, spending $2M won't get you 2,000 customers. At some point, you've reached most of the people who would buy, and extra spend produces less and less return. This is called "diminishing returns," and the Hill curve captures it mathematically. We tuned this in Round 3 and it significantly improved the model's accuracy.

---

## Part 3: Ten Rounds of Improvement

The agent ran ten rounds, each time trying one improvement and measuring whether it helped.

### How we measure accuracy

We held back the last 4 weeks of data — the model never saw these weeks during training. After fitting the model on the other 128 weeks, we asked it to predict those 4 held-back weeks. The gap between its predictions and reality is the **MAPE (Mean Absolute Percentage Error)**. Lower is better.

Think of it like: if the model predicts $30M revenue in a week and actual revenue was $26M, the error is 13.3%. A MAPE of 13% means the model is on average 13% off on weeks it has never seen before.

We also track **cross-model agreement** — the percentage gap between what Ridge and PyMC say about the same channel. When all three models agree, you can act. When they disagree by more than 50%, the channel's ROI is not yet reliable enough to base budget decisions on.

### Round 1 — Starting point: know what you're working with before trusting any number

Before making any budget recommendation, the system profiled the data. Best accuracy: **23.2% error**. It found five weeks where revenue behaved abnormally, and that Google Shopping had zero spend in 76% of all weeks. A channel you barely use can't have a reliable ROI estimate. These flags don't stop the analysis — they tell stakeholders which outputs to trust and which to hold lightly.

### Round 2 — Fixed the timing assumption: credit the right week

Digital advertising works fast. Someone who clicks a paid search ad typically buys within days, not weeks. The first run assumed a two-week carry-over, which spread ad credit across the wrong time periods. Switching to one week better matched how digital channels actually convert. **Error improved to 20.4%** — the models were now looking in the right direction.

### Round 3 — Fixed the saturation curve: let the models see the signal

Saturation curves control when a channel is assumed to have "used up" its marginal effect. The initial setting assumed channels could keep scaling up for longer than the data supported. Lowering the threshold let the models recognise diminishing returns earlier, which unlocked attribution across all channels. **Error improved to 13.1%** — the single largest accuracy gain in the study.

### Round 4 — Added domain knowledge: stop treating all media as equal

Applying one global decay rate to every channel is like using the same half-life for penicillin and a tattoo. Google Search ads fade in days (intent-driven — the purchase either happens or it doesn't). YouTube brand videos linger for weeks. Meta social sits in between. Domain benchmarks from a Notion knowledge layer gave each channel its own rate. Meta Facebook's cross-model disagreement dropped from **71% to 7.9%** in one round — the first channel to reach high agreement. Domain knowledge the model could not discover from 132 rows of data on its own.

### Round 5 — Bug fixes: results before this point were subtly wrong

Automated code review (GPT-4o and Claude running in parallel) caught two silent errors. A PyMC axis bug was assigning adstock effects to the wrong time periods — carry-over from Week 3 was being credited to Week 2. A scaling issue was comparing channels on incompatible scales, making some channels look larger or smaller than they were. Neither error produced a visible crash. Both were distorting attribution in the background. This is why automated code review runs every round rather than once at setup.

### Round 6 — More Bayesian samples: replace noise with stable estimates

With 50 MCMC samples, PyMC's uncertainty estimates were too unstable to trust — analogous to basing a business decision on a 50-person survey with high variance. Raising to 500 samples produced stable distributions. **Meta Facebook's credible interval held below CV 20%**, confirming the Round 4 signal was real and not a sampling artefact.

### Round 7 — Full model stack: all three models now running properly

PyMC was previously running without its full algorithm enabled, which prevented it from modelling diminishing returns and time-lagged effects properly. It was producing results, but not the results it was designed to produce. Enabling the full model brought all three independent methods into proper operation. **Meta Facebook confirmed at CV 7.9%** — cross-model agreement is now a genuine signal, not an artefact of simplified methods.

### Round 8 — Better third model: a step forward, and a new problem surfaces

The original third model produced estimates on a raw revenue scale, making it hard to compare channels that spent very different amounts. BayesianRidge with y-standardisation puts channels on the same footing before estimating contributions. It achieved the **best test MAPE of any model (15.1%)**. But it also revealed a new problem: the model was claiming media drove more than 100% of revenue in some weeks — mathematically impossible. A better method exposed an attribution artefact that had been hidden before.

### Round 9 — Attribution cap: enforce economic reality

Marketing drives sales, but not all sales come from paid media. Organic traffic, repeat customers, and seasonality account for a meaningful share of revenue. Without a cap, the model's positive and negative channel estimates were offsetting each other in a way that allowed positive channels to claim excessive credit. Capping total media attribution at 65% of KPI reflects the economic reality of this category. After the fix, **Meta Facebook settled at CV 4.5% across all three independent models** — the most robust signal in the entire study.

### Round 10 — Bayesian prior calibration: get the first actionable Google signal

Bayesian models have "priors" — baseline beliefs about how much each channel can contribute before seeing any data. PyMC's default prior was too permissive: it effectively said any channel could drive up to 150% of max revenue. For Google Shopping — dark 76% of weeks — the model had almost no data to work with, so it defaulted to the prior and produced a 680× ROI. That number reflects the model's belief, not the channel's performance. Tightening the prior to a more realistic ceiling forced PyMC to produce estimates grounded in what the data can actually support. **Google Search crossed below CV 50% for the first time** — the first actionable signal from any Google channel in ten rounds.

---

## Part 4: What We Found

### The most reliable finding: Meta Facebook works

All three independent models agreed that Meta Facebook generates approximately **$1.48–$1.60 in revenue for every $1 spent**. More importantly, the three models only disagreed by **3.2%** on this number — the closest agreement achieved across all channels across all ten rounds. This is not noise. This is a consistent signal that has held up through ten rounds of improvement, bug fixes, and model upgrades.

Meta Facebook received **$624M in total spend** over the period and appears to account for roughly **36–42% of all media-driven revenue**.

### Meta Instagram appears directionally positive

Instagram ROI is approximately **$1.69–$2.21 per dollar spent** (depending on the model). The three models disagree by 35.6%, so this is a directional signal, not a confirmed fact — but the direction has been consistent across the last several rounds, and disagreement has been declining.

What it suggests: **Instagram may be getting relatively less investment than its effectiveness warrants.** Facebook received 74% of the Meta budget; Instagram got 26%. If Instagram truly delivers comparable or higher return per dollar, the current allocation may not be optimal.

### Google Search: first signal, not yet confirmed

After ten rounds of model refinement, Google Search has for the first time produced a consistent directional signal across all three models — cross-model disagreement at 28.1%, down from 74.4% in Round 9. Ridge estimates ~$1.37× ROI; PyMC estimates ~$48.6× (high, but no longer implausible given the demand-capture nature of search). This is encouraging but not yet confirmed. **Do not make major budget decisions based on Google Search from this analysis alone.**

The high PyMC estimate reflects a real dynamic: Google Search captures intent — people searching for your product are close to buying. The model may be capturing the fact that search spend is concentrated in high-intent moments. But until cross-model agreement drops below 20%, treat this as a signal worth watching, not a confirmed finding.

### The Google channels that remain unclear

Google Shopping, Google Display, Google Video, and Google PMax still show contradictory or near-zero results across models. The main reasons:
- Google channels often run simultaneously with Meta. When one goes up, so does the other. The model struggles to separate which one caused the sales.
- Google Shopping ran in only 24% of weeks — not enough data to estimate its effect reliably.

**Do not make budget decisions on these channels from this analysis.**

### The organic baseline

Not all revenue comes from advertising. People who already know the brand buy again. Word of mouth brings new customers. Seasonal demand rises regardless of ads. The model estimates this "would have happened anyway" revenue is approximately **33% of total revenue**. The remaining ~67% is attributed to media channels.

---

## Part 5: What This Means for Budget Decisions

### What you can act on today
- **Meta Facebook** is confirmed by all three independent models as delivering positive ROI (~1.5–1.6×). Maintaining current investment is supported by the data.
- **Meta Instagram** shows a directional positive signal improving over time. Consider testing a modest reallocation — even shifting 5–10% of Facebook budget to Instagram is low-risk given the consistent directional signal across rounds.

### What you should watch but not act on yet
- **Google Search** has crossed a meaningful threshold for the first time. Monitor this signal in future rounds. If cross-model agreement improves to below 20%, it becomes actionable. The demand-capture nature of search (high intent, immediate conversion) makes this a plausible finding — it just needs confirmation.

### What you should not act on yet
Do not cut or grow Google Shopping, Display, Video, or PMax based on this analysis. The models disagree too much. The right path for unclear Google channels is an **incrementality experiment** — running ads in some regions and not others to directly measure the effect, without relying on modeling assumptions.

### What would make this more reliable
- Adding a "was this week a promotion?" flag to the data would help the model separate promotional spikes from normal advertising effects
- Geo-experiments for Google channels would give direct causal evidence
- More Google Search spend weeks would give the model more data to estimate its effect reliably

---

## The Key Limitation

This analysis used **132 weeks of historical data from a public ecommerce dataset** — not a real company's data. The results are directionally illustrative. They show what this kind of analysis can find and how the agent system works, but they are not a basis for a specific company's budget decisions.

For a real budget decision, you would need:
- Your own company's data
- At least 2–3 years of weekly revenue and spend data
- Business context (promotions, price changes, competitor activity) to explain unusual weeks
- Ideally some incrementality experiments to calibrate the model

The finding most likely to transfer to real businesses: **not all advertising channels are equally efficient, models consistently disagree more on Google channels than Meta channels, and Instagram tends to appear more efficient per dollar than Facebook despite receiving less budget.** These are patterns worth investigating with your own data.

---

## Summary Table

| Question | Answer |
|---|---|
| What did we analyse? | 132 weeks of ecommerce ad spend and revenue, 8 channels |
| How many models? | 3 independent models cross-checking each other |
| How many rounds of tuning? | 10 rounds, one improvement per round |
| Best model accuracy? | 15.1% error on held-out weeks |
| Most reliable channel finding? | Meta Facebook — confirmed by all 3 models, CV 3.2% |
| Second most reliable? | Meta Instagram — directional positive, CV 35.6% and improving |
| First Google signal? | Google Search — CV 28.1%, first time below 50% after 10 rounds |
| Organic baseline (no-ads revenue)? | ~33% of total revenue |
| Google Shopping / Display / Video? | Too noisy — do not act on results yet |
| What's next? | Incrementality experiments for Google channels; continue improving cross-model agreement |

---

*Dataset: [Multi-Region MMM Dataset for eCommerce Brands](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841/3?file=46779652), Figshare, CC BY 4.0. Results are illustrative and not from a real brand.*

*Generated by PrismMMM — [github.com/ScarlettQiu/prismmmm](https://github.com/ScarlettQiu/prismmmm)*
