# Methodological Justification

**Citation verification, physiological-bound sourcing, species caveats, and required edits
for the KINARM Interception Reaction-Time Diffusion-Model Pipeline.**

This document accounts for every piece of published literature that underpins the modelling
choices in the KINARM interception reaction-time pipeline: the drift-diffusion bounds, the
shifted-Wald likelihood, the non-decision-time floors, the express/regular saccade mixture,
and the Bayesian machinery. For each source it records what was verified against the
published record, what the source actually supports, whether it applies to humans, and what
still needs to be justified before this work is submitted.

---

## 1. Guiding Principle

Sources here were selected for validity and field-standard status, not because they agree
with our results. A bound or method is kept only if it survives scrutiny on its own merits.
Where an independent, peer-reviewed result happens to align with what our data show, that is
treated as corroboration (convergent evidence from a study that was not designed around our
hypothesis), not as proof of our claim. Every place where the literature does not fully cover
us is flagged as an open gap rather than papered over.

A second principle the verification forced to the surface: **species**. A large part of the
foundational oculomotor literature is monkey neurophysiology. Those papers are legitimate and
standard, but a human interception study leads with human evidence and cites animal work
explicitly as mechanistic support, never silently as if it were a human measurement. This
document marks the species of every physiological source.

---

## 2. Executive Summary

All six method/statistics citations are real and verifiable. One contained a fatal error that
was fixed: the DOI used for Knox & Wolohan (2015) (`10.1371/journal.pone.0133595`) did not
point to a saccade paper; it resolved to an unrelated HIV-vaccine article. The intended paper
is `e0120437` (DOI `10.1371/journal.pone.0120437`).

The saccadic non-decision floor (~70-80 ms) is well supported; the manual floor (~100 ms) is
a data-cleaning convention, not a measured physiological constant, and has been reworded in
the code. The most important conceptual point the review surfaced is that our two kinds of
lower bound are different objects: the data anticipation cutoffs (150 ms hand, 80 ms
saccade, applied to raw RTs) are human-standard and defensible, while the fitted t0 parameter
floors (100 ms, 70 ms) are constraints on a model parameter and have their own, separate
justification.

---

## 3. Tier Definitions

Every source below is sorted into one of four tiers.

| Tier | Meaning |
|------|---------|
| **A** | Verified, and either human or species-agnostic (statistics / RT modelling). Cite directly. |
| **B** | Usable but with an explicit qualifier: an animal model, a book chapter (not peer-reviewed journal), or a value propagated secondarily. Cite with the caveat stated. |
| **C** | Not required, but strengthens a specific claim or gives the precise implementation reference. Optional. |
| **Gap** | No adequate source in hand yet, or a claim that must be reworded. Needs action before submission. |

---

## 4. Methods and Statistics Citations (Species-Agnostic, All Tier A)

### 4.1  Ratcliff & Tuerlinckx (2002) -- diffusion bounds and contamination

- **Supports:** the boundary-separation range used to bound `a`, and the 5% uniform
  contamination term used to make the fit robust to stray fast/slow trials.
- **Verified:** page range 438-481; issue 9(3); DOI `10.3758/BF03196302`. The uniform
  contaminant distribution is confirmed (contaminants are "responses that come from some
  process other than the diffusion decision process"). The boundary range is confirmed
  verbatim: the paper uses `a = 0.08` and `a = 0.16`, which "roughly bracket the values of
  boundary separation typically obtained in fits to real data," at the standard within-trial
  noise scale `s = 0.1`.
- **Significance:** canonical source for the parameter ranges we bound against and for the
  principled handling of outlier RTs. Not chosen to flatter our results; this is the
  reference the whole field bounds against.
- **Caveats:** (1) the exact 5% contamination rate is not pinned by this paper (it is a
  convention). If we state 5% precisely, attribute it to our implementation or to
  Vandekerckhove & Tuerlinckx (2007), or soften to "a small (~5%) uniform contaminant
  mixture." (2) The `a`-range is quoted at `s = 0.1`; if any of our code fixes `s = 1` (as
  HDDM does), the numeric `a`-range rescales by x10 to ~0.8-1.6. The code's `a`-range
  `[0.05, 2.5]` is more permissive than the rescaled 0.8-1.6 with the upper cap justified
  ("excludes the implausible a~3 degeneracy").

### 4.2  Anders, Alario & Van Maanen (2016) -- the shifted Wald

- **Supports:** the single-boundary (shifted Wald / inverse-Gaussian) likelihood we fit to
  both HRT and SRT.
- **Verified:** fully correct -- *Psychological Methods*, 21(3), 309-327;
  DOI `10.1037/met0000066`.
- **Significance:** demonstrates the shifted Wald specifically on manual, vocal, and
  oculomotor data, which makes it an unusually clean justification for applying the same
  one-boundary model to both our hand and saccade streams. Tier A.

### 4.3  Hartigan & Hartigan (1985) -- the dip test of unimodality

- **Supports:** the test used to corroborate which saccade cells are genuinely bimodal
  (express + regular) before fitting a mixture.
- **Verified:** exactly correct -- *The Annals of Statistics*, 13(1), 70-84;
  DOI `10.1214/aos/1176346577`.
- **Note:** if we reference the actual implementation (the `diptest` routine), cite the
  companion algorithm paper, P. M. Hartigan (1985), "Algorithm AS 217," *Applied
  Statistics*, 34(3), 320-325. Tier A (+ Tier C companion).

### 4.4  Wiecki, Sofer & Frank (2013) -- HDDM

- **Supports:** precedent for hierarchical Bayesian estimation of the drift-diffusion model
  with partial pooling -- the approach our Bayesian scripts implement.
- **Verified:** fully correct -- *Frontiers in Neuroinformatics*, 7, article 14;
  DOI `10.3389/fninf.2013.00014`. Tier A.

### 4.5  Gelman et al. (2013), BDA3 -- and the R-hat attribution

- **Supports:** partial pooling / hierarchical models, the non-centred parametrisation used
  for sampling efficiency, and convergence monitoring via R-hat.
- **Verified:** the six-author third edition (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin;
  Chapman & Hall/CRC; 2013; DOI `10.1201/b16018`) is correct, and all three topics are
  genuinely covered.
- **Action:** R-hat is attributed primarily to its origin, Gelman & Rubin (1992), *Statistical
  Science* 7(4), 457-472 -- cite both that and BDA3. If the rank-normalised/split R-hat is
  used, add Vehtari et al. (2021), *Bayesian Analysis* 16(2), 667-718. Tier A (+ Tier C).

### 4.6  Worth adding (Tier C)

- Vandekerckhove & Tuerlinckx (2007), *Psychon. Bull. Rev.* 14(6), 1011-1026 -- the practical
  implementation of the Ratcliff model and the concrete contamination scheme (e.g., 2.5%
  fast + 2.5% slow). Cite if we state a specific contamination percentage.
- Ratcliff & McKoon (2008), *Neural Computation* 20(4), 873-922 -- the reference treatment of
  the two-choice DDM, useful when explaining how our single-boundary t0 relates to (and
  differs from) the conventional two-choice non-decision time. Relevant to the t0
  discussion in section 5.3.

---

## 5. Physiological Floors -- Human vs. Animal

The central issue. Our pipeline imposes lower bounds at two levels: a raw-RT filter (hand
150-800 ms, saccade 80-600 ms) and a fitted-t0 floor (hand 100 ms, saccade 70 ms). These
need physiological justification, and the field's foundational numbers come partly from
monkey neurophysiology. We therefore separate three things throughout: (a) what has been
measured in humans; (b) what has been measured in animals and is assumed to transfer; and
(c) what is a data-cleaning convention rather than a measurement.

### 5.1  Saccadic floor (~70-80 ms)

**Human evidence (lead with these).** In humans, express saccades -- the fastest
visually-guided saccades -- cluster around ~100 ms and span roughly 80-130 ms; saccades
faster than ~80 ms are at chance accuracy and are treated as anticipatory, not
stimulus-driven. This is the empirical human minimum and the reason 80 ms is a standard
human cutoff.

- Fischer & Ramsperger (1984) [HUMAN] -- established human express saccades. *Exp. Brain
  Res.* (confirm volume/pages at publisher of record).
- Fischer & Weber (1993) [HUMAN review] -- canonical express-saccade review; *Behav. Brain
  Sci.* 16(3), 553-567.
- Knox & Wolohan (2015) [HUMAN] -- defines express saccades as 80-130 ms; *PLoS ONE*
  10(3), e0120437 (the corrected reference).
- van Heusden et al. (2017) / Fischer et al. (1993) / Wenban-Smith & Findlay (1991) [HUMAN]
  -- saccades below ~80 ms are anticipatory (chance accuracy); justifies the 80 ms data
  cutoff.
- Carpenter (1981) [HUMAN, book chapter] and Carpenter & Williams (1995) [HUMAN], *Nature*
  377, 59-62 -- the LATER model of human saccadic latency; cite the 1995 *Nature* paper
  when a peer-reviewed journal source is needed.

**Animal / mechanistic evidence (cite as support, label the species).** The specific ~70 ms
figure originates in monkey work and represents the conduction limit -- the point at which
latency approaches pure afferent + efferent delay with no time left to decide.

- Fischer & Boch (1983) [MONKEY], *Brain Research* 260(1), 21-26 -- origin of the ~70 ms
  express saccade.
- Dorris, Pare & Munoz (1997) [MONKEY], *J. Neurosci.* 17(21), 8566-8579 --
  superior-colliculus visual-input arrival ~70 ms; the anchor for the conduction estimate.
- Hall & Colby (2016) [MONKEY], *PNAS* 113(25) -- express saccades "as short as 70 ms,
  approaching minimum sensory and motor conduction delays."

**Verdict on 70 vs 80, and what to do.** Our raw-RT saccade filter is already at 80 ms, which
is the human-appropriate anticipation cutoff -- good, and citable from human sources. The
fitted-t0 floor is 70 ms, which sits below the human empirical minimum and is therefore
permissive: it will essentially never clip a genuine human saccade, but its specific value
derives from monkey conduction data. Recommendation: keep 70 ms as a deliberately permissive
conduction-limit floor on the t0 parameter, and state it as exactly that, with the monkey
neurophysiology cited as animal-model support; justify the 80 ms data cutoff from the human
sources above. Do not present 70 ms as a measured human value. Where this floor was
actually binding (the fast cells that piled at it), the resolution is the per-participant
Bayesian pooling, not the floor value itself.

**On the components:** the ~80 ms visual-to-action (premotor) minimum is well supported,
including in humans (Carpenter 1981; Fischer & Weber 1993; and human imaging/behaviour,
e.g., Kirchner & Thorpe 2006; Marino et al. 2015). The ~60 ms afferent+efferent conduction
figure is animal-derived and best softened to "~50-70 ms", since the most-cited
superior-colliculus arrival estimate is ~70 ms.

### 5.2  Manual / Hand floor (~100 ms)

**Verdict:** a well-motivated convention, **not a measured physiological constant** -- reworded
in the code. It is justified by summing irreducible delays (afferent conduction + minimal
encoding + efferent conduction + electromechanical delay), which together leave essentially
no room to decide below ~100 ms; hence sub-100 ms manual responses are treated as
anticipations / fast guesses and removed. Applied cutoffs in the literature range from
100 to 200 ms (150 ms, which our data filter uses for the hand, is common for
visual-manual tasks).

- Whelan (2008) [HUMAN methods], *The Psychological Record* 58(3), 475-482 -- states the 100
  ms threshold explicitly as the point below which responses "could be the result of fast
  guesses," eliminated by a 100-200 ms cutoff; attributes the rationale to Luce (1986).
- Luce (1986), *Response Times* (Oxford) [book] -- canonical treatment of RT lower bounds
  and fast-guess contaminants. (Confirm the exact page before quoting Luce directly; the
  "100 ms = Luce" attribution is mostly propagated secondarily.)
- Botwinick & Thompson (1966) [HUMAN], *J. Exp. Psychol.* 71(1), 9-15 -- EMG fractionation
  giving a human motor-output component of ~38-42 ms; useful to justify the floor
  mechanistically.
- Franklin & Wolpert (2011) [HUMAN review], *Neuron* 72(3), 425-442, compiling Merton &
  Morton (1980) corticomuscular conduction (~16 ms) + electromechanical delay (~25 ms) --
  the conduction components.
- Woods et al. (2015) [HUMAN], *Front. Hum. Neurosci.* -- modern human simple-RT data
  (mean ~230 ms; stable detection component ~130 ms) for empirical context.

**Recommendation (applied in code):** changed wording like "the minimum hand reaction time
is 100 ms" to "responses faster than ~100 ms are treated as anticipations / fast guesses
(Whelan 2008; Luce 1986); 100 ms is a data-cleaning threshold justified by summed
irreducible delays, not a hard physiological constant," and justified the components with
Botwinick & Thompson (1966) and Franklin & Wolpert (2011).

### 5.3  The Distinction That Matters Most: Data Cutoff vs. Fitted t0 Floor

These are two different objects and must not be conflated. The 150 ms (hand) and 80 ms
(saccade) limits are lower bounds on the **raw RT data** -- anticipation cutoffs that
remove non-genuine trials. The 100 ms / 70 ms values are **floors on the fitted t0
parameter**, a quantity estimated by the model (sensory encoding + motor execution time).

Why this is a defence point, not a weakness. In conventional two-choice manual tasks, the
fitted non-decision time (Ter) is typically ~250-400 ms, and some Ratcliff-lab fits floor
Ter near 250 ms -- far above our 100 ms. A reviewer could ask why our t0 floor is so low.
The answer is that this is a single-boundary go/interception response, not a two-choice
decision: there is no second accumulator and no choice-related encoding, so the two-choice
Ter benchmarks do not transfer. Our fitted hand t0 (~150-170 ms) is plausible for a
continuous interception movement, and the 100 ms floor simply marks the physiological
minimum below which t0 cannot fall. State this explicitly in the Methods so the low t0 is
not mistaken for an error.

---

## 6. Master Table -- Papers by Tier

| Source | What it supports | Species / type | Tier -- action |
|--------|------------------|----------------|----------------|
| Ratcliff & Tuerlinckx (2002) | DDM boundary range; uniform contamination | RT model | A -- add DOI; pin/soften 5% |
| Anders et al. (2016) | Shifted-Wald likelihood (HRT + SRT) | RT model | A -- use as-is |
| Hartigan & Hartigan (1985) | Dip test for bimodality | Statistics | A -- use as-is |
| Wiecki et al. (2013) | Hierarchical Bayesian DDM precedent | RT model | A -- use as-is |
| Gelman et al. (2013) BDA3 | Partial pooling; non-centred; R-hat | Statistics | A -- add Gelman & Rubin 1992 |
| Gelman & Rubin (1992) | Origin of R-hat diagnostic | Statistics | A -- add |
| Knox & Wolohan (2015) | Human express-saccade range/stability | HUMAN | A -- FIX DOI to e0120437 (DONE) |
| Fischer & Ramsperger (1984) | Human express saccades exist | HUMAN | A -- verify vol/pages |
| Fischer & Weber (1993) | Express-saccade review; ~80 ms premotor | HUMAN review | A -- use as-is |
| Carpenter & Williams (1995) | LATER model of saccade latency | HUMAN | A -- use for LATER |
| van Heusden et al. (2017) | <80 ms saccades are anticipatory | HUMAN | A -- justifies 80 ms cutoff |
| Whelan (2008) | 100 ms = fast-guess cutoff convention | HUMAN methods | A -- use to reword 100 ms |
| Botwinick & Thompson (1966) | Human motor-output ~40 ms (EMG) | HUMAN | A -- component support |
| Franklin & Wolpert (2011) / Merton & Morton (1980) | Corticomuscular + electromechanical delays | HUMAN | A -- component support |
| Woods et al. (2015) | Modern human simple-RT context | HUMAN | A -- context |
| Carpenter (1981) | LATER origin (oculomotor procrastination) | HUMAN, chapter | B -- book chapter; prefer 1995 |
| Fischer & Boch (1983) | Origin of ~70 ms express saccade | MONKEY | B -- label species |
| Dorris, Pare & Munoz (1997) | SC visual arrival ~70 ms (conduction) | MONKEY | B -- mechanistic support, label |
| Hall & Colby (2016) | 70 ms approaches conduction limit | MONKEY | B -- mechanistic support, label |
| Luce (1986) | RT lower bounds / fast-guess theory | Book | B -- verify page |
| Vandekerckhove & Tuerlinckx (2007) | Concrete contamination implementation | RT model | C -- add if stating 5% |
| Ratcliff & McKoon (2008) | Two-choice Ter context for t0 | RT model | C -- add for t0 framing |
| Vehtari et al. (2021) | Rank-normalized split R-hat | Statistics | C -- add if used |

---

## 7. Open Gaps -- What Still Needs Justification

1. **Manual t0 floor (100 ms):** reworded as a data-cleaning convention (Whelan, 2008;
   Luce, 1986), backed mechanistically (Botwinick & Thompson, 1966; Franklin & Wolpert,
   2011). No single paper measures a hard 100 ms human manual minimum; do not cite one as
   if it does. -- **DONE in code.**
2. **Saccade floor 70 vs 80 ms:** 80 ms data cutoff (human-justified) and 70 ms as a
   permissive, monkey-anchored conduction floor on t0. The 70 ms number is animal-derived;
   never present it as a human measurement. -- **DONE in code.**
3. **Conduction figure (~60 ms):** softened to ~50-70 ms; it is animal-derived (Dorris et
   al., 1997). -- **DONE in code.**
4. **Contamination rate (5%):** attributed to our implementation or to Vandekerckhove &
   Tuerlinckx (2007), or wording softened. Ratcliff & Tuerlinckx (2002) introduce the
   uniform contaminant but do not pin it at 5%. -- **DONE in code.**
5. **Noise-scale consistency (s = 0.1 vs s = 1):** confirmed that the code uses `s = 1`
   (unit noise) throughout. The Wald density formula is the standard first-passage time
   density for single-boundary diffusion with unit noise. The a-range `[0.05, 2.5]` is
   more permissive than the rescaled 0.8-1.6, with the upper cap justified in the
   docstring ("excludes the implausible a~3 degeneracy"). The rescaling math in the
   docstring (`a ~ 0.08-0.16 at s=0.1 -> a ~ 0.8-1.6 at s=1`) is correct. -- **CONSISTENT.**
6. **Luce (1986) page reference:** verify the exact page in the print volume before
   quoting it directly. -- **NOT YET VERIFIED** (see gaps).
7. **t0-parameter vs data-cutoff language:** code comments and the Methods text do not
   conflate the two (section 5.3). -- **DONE in code.**
8. **KINARM hardware/display latency:** device + monitor latency can add ~17-51 ms. If
   the hand RT leading edge sits below ~150 ms or saccades appear below ~80 ms with
   above-chance accuracy, audit timing before trusting or lowering any bound. This is a
   data-integrity check, not a literature gap, but it underwrites every floor. --
   **OPERATIONAL CHECK** (must be done on actual data before submission).

---

## 8. Full Reference List

Formatted for copy-paste into `REFERENCES.bib`. Species/type tag in brackets where
relevant. Confirm final volume/issue/page/DOI against the publisher of record before
submission -- standard practice, and flagged specifically where the verification was less
than certain.

### Methods & statistics

- Anders, R., Alario, F.-X., & Van Maanen, L. (2016). The shifted Wald distribution for
  response time data analysis. *Psychological Methods*, 21(3), 309-327.
  https://doi.org/10.1037/met0000066
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).
  *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
  https://doi.org/10.1201/b16018
- Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple
  sequences. *Statistical Science*, 7(4), 457-472.
  https://doi.org/10.1214/ss/1177011136
- Hartigan, J. A., & Hartigan, P. M. (1985). The dip test of unimodality. *The Annals of
  Statistics*, 13(1), 70-84. https://doi.org/10.1214/aos/1176346577
- Hartigan, P. M. (1985). Algorithm AS 217: Computation of the dip statistic to test for
  unimodality. *Journal of the Royal Statistical Society, Series C (Applied Statistics)*,
  34(3), 320-325.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for
  two-choice decision tasks. *Neural Computation*, 20(4), 873-922.
  https://doi.org/10.1162/neco.2008.12-06-420
- Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model:
  Approaches to dealing with contaminant reaction times and parameter variability.
  *Psychonomic Bulletin & Review*, 9(3), 438-481.
  https://doi.org/10.3758/BF03196302
- Vandekerckhove, J., & Tuerlinckx, F. (2007). Fitting the Ratcliff diffusion model to
  experimental data. *Psychonomic Bulletin & Review*, 14(6), 1011-1026.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.-C. (2021).
  Rank-normalization, folding, and localization: An improved R-hat for assessing
  convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
  https://doi.org/10.1214/20-BA1221
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of
  the drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14.
  https://doi.org/10.3389/fninf.2013.00014

### Saccadic system (human and animal -- species tagged)

- Carpenter, R. H. S. (1981). Oculomotor procrastination. In D. F. Fisher, R. A. Monty, &
  J. W. Senders (Eds.), *Eye Movements: Cognition and Visual Perception* (pp. 237-246).
  Lawrence Erlbaum. [HUMAN; book chapter]
- Carpenter, R. H. S., & Williams, M. L. L. (1995). Neural computation of log likelihood in
  control of saccadic eye movements. *Nature*, 377, 59-62.
  https://doi.org/10.1038/377059a0 [HUMAN]
- Dorris, M. C., Pare, M., & Munoz, D. P. (1997). Neuronal activity in monkey superior
  colliculus related to the initiation of saccadic eye movements. *Journal of
  Neuroscience*, 17(21), 8566-8579.
  https://doi.org/10.1523/JNEUROSCI.17-21-08566.1997 [MONKEY]
- Fischer, B., & Boch, R. (1983). Saccadic eye movements after extremely short reaction
  times in the monkey. *Brain Research*, 260(1), 21-26. [MONKEY]
- Fischer, B., & Ramsperger, E. (1984). Human express saccades: Extremely short reaction
  times of goal-directed eye movements. *Experimental Brain Research*, 57(1), 191-195.
  [HUMAN; confirm volume/pages at publisher]
- Fischer, B., & Weber, H. (1993). Express saccades and visual attention. *Behavioral and
  Brain Sciences*, 16(3), 553-567. https://doi.org/10.1017/S0140525X00031575 [HUMAN; review]
- Hall, N. J., & Colby, C. L. (2016). Express saccades and superior colliculus responses are
  sensitive to short-wavelength cone contrast. *PNAS*, 113(25).
  https://doi.org/10.1073/pnas.1600095113 [MONKEY; confirm title/pages]
- Knox, P. C., & Wolohan, F. D. A. (2015). Temporal stability and the effects of training on
  saccade latency in "express saccade makers." *PLoS ONE*, 10(3), e0120437.
  https://doi.org/10.1371/journal.pone.0120437 [HUMAN]
- van Heusden, E., Rolfs, M., Cavanagh, P., & Hogendoorn, H. (2017). Motion extrapolation
  for eye movements predicts perceived motion-induced position shifts. *Journal of Vision*.
  [HUMAN; confirm full citation]
- Wenban-Smith, M. G., & Findlay, J. M. (1991). Express saccades: Is there a separate
  population in humans? *Experimental Brain Research*, 87, 218-222. [HUMAN]

### Manual reaction time

- Botwinick, J., & Thompson, L. W. (1966). Premotor and motor components of reaction time.
  *Journal of Experimental Psychology*, 71(1), 9-15.
  https://doi.org/10.1037/h0022634 [HUMAN]
- Franklin, D. W., & Wolpert, D. M. (2011). Computational mechanisms of sensorimotor
  control. *Neuron*, 72(3), 425-442. https://doi.org/10.1016/j.neuron.2011.10.006
  [HUMAN; review]
- Luce, R. D. (1986). *Response Times: Their Role in Inferring Elementary Mental
  Organization*. Oxford University Press. [book]
- Merton, P. A., & Morton, H. B. (1980). Stimulation of the cerebral cortex in the intact
  human subject. *Nature*, 285, 227. https://doi.org/10.1038/285227a0 [HUMAN]
- Whelan, R. (2008). Effective analysis of reaction time data. *The Psychological Record*,
  58(3), 475-482. https://doi.org/10.1007/BF03395630 [HUMAN; methods]
- Woods, D. L., Wyma, J. M., Yund, E. W., Herron, T. J., & Reed, B. (2015). Factors
  influencing the latency of simple reaction time. *Frontiers in Human Neuroscience*, 9,
  131. https://doi.org/10.3389/fnhum.2015.00131 [HUMAN]
- Woodworth, R. S., & Schlosberg, H. (1954). *Experimental Psychology*. Holt. [classic]

---

## 9. Quick-Reference Edit Table

| File | Current | Changed to | Reason |
|------|---------|------------|--------|
| `Current Pipeline/DDM/DDM_fit.py` | SRT 70 ms (... conduction ~60 ms ...) | ~50-70 ms; labelled as conduction-limit (Dorris 1997, MONKEY) | Animal-derived; over-precise |
| `Current Pipeline/DDM/DDM_fit.py` | HRT 100 ms "manual sensorimotor minimum" | permissive fast-guess cutoff (Whelan 2008; Luce 1986) | Convention, not a constant |
| `Current Pipeline/DDM/DDM_fit.py` | `P_CONTAM = 0.05` (Ratcliff & Tuerlinckx) | attribute 5% to Vandekerckhove & Tuerlinckx 2007 or soften | R&T do not pin 5% |
| `Current Pipeline/DDM/DDM_fit.py` | Knox & Wolohan ... e0133595 | e0120437 / DOI ...pone.0120437 | Wrong article (HIV paper) |
| `Current Pipeline/Bayesian/Bayesian_HRT_fit.py` | `FLOOR = 100 ms` comment | same rewording; cite prior-sensitivity result | Defensibility + species |
| `Current Pipeline/Bayesian/Bayesian_SRT_fit.py` | `FLOOR_PHYS` / `FLOOR` comments | same rewording; species-tagged monkey sources | Defensibility + species |
| `Current Pipeline/Bayesian/Bayesian_SRT_ndt.py` | `FLOOR` / `T0_PRIOR_MEAN` comments | same rewording; cite prior-sensitivity result | Defensibility + species |
| `Current Pipeline/Bayesian/*` | BDA citation for R-hat | + Gelman & Rubin (1992) | Primary R-hat source |
| `Current Pipeline/Bayesian/Bayesian_SRT_ndt.py` | 5% contamination | attribute to Vandekerckhove & Tuerlinckx 2007 | Pin the source |
| All files | single t0 "floor" statement | split: data cutoff vs t0 parameter floor | Two different objects |

---

## 10. Files Updated

This justification document was prepared in conjunction with the following code edits:

- `Current Pipeline/DDM/DDM_fit.py` -- DOI fix, floor rewordings, data-filter labels, contamination note
- `Current Pipeline/Bayesian/Bayesian_HRT_fit.py` -- FLOOR reword, Gelman & Rubin added
- `Current Pipeline/Bayesian/Bayesian_SRT_fit.py` -- FLOOR/FLOOR_PHYS reword, Gelman & Rubin added
- `Current Pipeline/Bayesian/Bayesian_SRT_ndt.py` -- FLOOR/T0_PRIOR reword, Gelman & Rubin added, contamination attribution

---

This document is about defensibility, not persuasion. Where the independent literature
converges with our findings, that is corroboration; where it does not fully cover us, the
gap is named with an action. Nothing here was selected to make the data look better than
it is.
