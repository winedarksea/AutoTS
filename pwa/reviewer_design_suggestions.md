
We need a proper theme for the forecasting pwa, which currently has none.

Generally we like Material Design 3 design patterns. We definitely want good support for both light and dark themes. Blue seems a bit overused, so we would lean towards using a cyan or turquoise instead of regular blue, potentially alongside a "wine dark sea" deep sea blue as well. A good color theme might perhaps be "ancient Greek" with colors of Homer's Odyssey, but while still aiming for a modern theme. Likely the overall impression should be "classical" (Roman mosaics and architecture, Grecian coastal landscape, in particular real world materials, particularly metallics, alongside stone) mixed with "modern" (Material 3 Expressive). A classical feel should pervade this app, but practical app needs (realized in modern styles) ultimately are the most important.

## Reviewer One Design Proposals:

A strong direction would be: Aegean cyan as the interactive brand color, wine-dark deep sea as the authority/structure color, stone as the surface system, bronze/gold as sparse emphasis, and terracotta/error as semantic warmth. The key is to encode this as Material 3 role tokens, not as ad hoc hex values sprinkled through CSS.

Material 3 is a good fit because its color model is role-based: primary, secondary, tertiary, error, surface, and outline groups, with light/dark variants and “on-*” foreground roles for contrast. Google’s current M3 guidance describes 26 standard color roles, and M3 Expressive extends the system with more vibrant color, shape contrast, motion, and emphasized typography.

1. Theme concept: “Aegean Classical Modern”

Avoid making the app look like a themed restaurant. The classical layer should mostly come from materials, ratios, texture, iconography, and restrained accent color, while the actual UI remains clean and modern.

A practical translation:

Theme source	UI translation
Aegean / turquoise water	Primary action color
“Wine-dark sea”	Dark structural color, hero backgrounds, dark theme surface tint
Marble / limestone / plaster	Neutral surfaces
Bronze / aged gold	Sparse premium accent, highlights, badges
Terracotta / amphora clay	Warning/negative/warm tertiary accent
Mosaics	Optional geometric separators, charts, empty states, illustrations
Greek/Roman architecture	Grid discipline, generous spacing, strong vertical rhythm
Material 3 Expressive	Rounded components, motion, color containers, high-emphasis typography

The app should feel like forecasting in an Aegean observatory, not like a blue SaaS dashboard.

2. Suggested seed colors

Start with three seed colors, then generate tonal palettes from them.

/* Brand seeds, not final role colors */
--seed-primary-aegean: #00A6A6;    /* cyan / turquoise */
--seed-secondary-sea:  #183A4A;    /* wine-dark deep sea blue */
--seed-tertiary-bronze:#B8792F;    /* bronze / ochre */
--seed-warm-clay:      #B85C38;    /* terracotta, use sparingly */

I would not use bright cyan as the whole brand surface. Use it for interaction and active states. Let deep sea and stone do most of the visual mass.

3. Map the concept into Material roles

A useful starting mapping:

M3 role	Recommended semantic use	Theme color family
primary	Main CTAs, selected nav, active controls	Aegean cyan
primary-container	Soft selected states, chips, highlighted forecast cards	Pale cyan
secondary	Structural controls, app chrome, secondary buttons	Deep sea
secondary-container	Dense panels, nav rail backgrounds, neutral callouts	Muted sea blue
tertiary	Rare emphasis, credibility badges, “expert mode” accents	Bronze/gold
tertiary-container	Soft warm panels, onboarding, explanatory callouts	Sand/ochre
surface	Main app background	Stone / marble neutral
surface-container-*	Cards, sidebars, elevated panels	Slightly shifted stone tones
outline	Borders, separators, chart gridlines	Cool stone gray
error	Forecast failure, invalid input, destructive actions	Terracotta/red-brown

Do not make tertiary the error color. Bronze should feel like classical artifact / priority / insight, not failure.

4. Light theme draft

This is a usable first pass. You should run these through a contrast checker and ideally generate proper HCT/Tonal Palette outputs with Material tooling, but the semantic structure is sound.
```
:root {
  color-scheme: light;

  /* Core surfaces */
  --md-sys-color-background: #fbf8f1;
  --md-sys-color-on-background: #1d1b17;

  --md-sys-color-surface: #fbf8f1;
  --md-sys-color-on-surface: #1d1b17;

  --md-sys-color-surface-container-lowest: #ffffff;
  --md-sys-color-surface-container-low: #f6f1e7;
  --md-sys-color-surface-container: #eee8dc;
  --md-sys-color-surface-container-high: #e8e0d3;
  --md-sys-color-surface-container-highest: #ddd4c5;

  /* Primary: Aegean cyan */
  --md-sys-color-primary: #006a6a;
  --md-sys-color-on-primary: #ffffff;
  --md-sys-color-primary-container: #7ff4ee;
  --md-sys-color-on-primary-container: #002020;

  /* Secondary: wine-dark sea */
  --md-sys-color-secondary: #365666;
  --md-sys-color-on-secondary: #ffffff;
  --md-sys-color-secondary-container: #d1ebf7;
  --md-sys-color-on-secondary-container: #071f2a;

  /* Tertiary: bronze / ochre */
  --md-sys-color-tertiary: #79511f;
  --md-sys-color-on-tertiary: #ffffff;
  --md-sys-color-tertiary-container: #ffddb0;
  --md-sys-color-on-tertiary-container: #2a1700;

  /* Error: clay-red */
  --md-sys-color-error: #9c4230;
  --md-sys-color-on-error: #ffffff;
  --md-sys-color-error-container: #ffdad3;
  --md-sys-color-on-error-container: #3d0600;

  /* Lines */
  --md-sys-color-outline: #81766a;
  --md-sys-color-outline-variant: #d2c7b8;
}
```
5. Dark theme draft

Dark mode should not be “black with neon cyan.” That would push the style toward cyberpunk. For this concept, use deep sea + dark stone + muted cyan glow.
```
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;

    --md-sys-color-background: #111719;
    --md-sys-color-on-background: #e7e1d8;

    --md-sys-color-surface: #111719;
    --md-sys-color-on-surface: #e7e1d8;

    --md-sys-color-surface-container-lowest: #0b1012;
    --md-sys-color-surface-container-low: #171d20;
    --md-sys-color-surface-container: #1c2428;
    --md-sys-color-surface-container-high: #263136;
    --md-sys-color-surface-container-highest: #303c42;

    --md-sys-color-primary: #5fdad5;
    --md-sys-color-on-primary: #003737;
    --md-sys-color-primary-container: #005050;
    --md-sys-color-on-primary-container: #8ff4ee;

    --md-sys-color-secondary: #b5cedb;
    --md-sys-color-on-secondary: #203641;
    --md-sys-color-secondary-container: #2f4c5b;
    --md-sys-color-on-secondary-container: #d1ebf7;

    --md-sys-color-tertiary: #eabf81;
    --md-sys-color-on-tertiary: #442b05;
    --md-sys-color-tertiary-container: #5d3d10;
    --md-sys-color-on-tertiary-container: #ffddb0;

    --md-sys-color-error: #ffb4a7;
    --md-sys-color-on-error: #5f160b;
    --md-sys-color-error-container: #7d2c1f;
    --md-sys-color-on-error-container: #ffdad3;

    --md-sys-color-outline: #9c9184;
    --md-sys-color-outline-variant: #51483f;
  }
}
```
6. Use cyan for “action,” sea-blue for “frame”

This distinction matters.

Use cyan for:

primary buttons
focused inputs
active tabs
selected forecast
live probability edits
“add forecast” / “resolve” / “submit prediction”

Use wine-dark sea for:

headers
navigation rail/sidebar
dark hero sections
large blocks of structure
footer
app shell
secondary buttons

Use bronze for:

accuracy badges
calibration insights
“high confidence” ornament
active streaks
model-quality indicators
sparse icon emphasis

Use stone neutrals for almost everything else.

7. Data visualization should have its own palette

Forecasting apps live or die on charts. Do not force all chart series through the brand palette.

Separate UI theme tokens from data-viz tokens:

:root {
  --viz-positive: #00796b;
  --viz-negative: #b4472f;
  --viz-neutral: #6f6a60;
  --viz-uncertainty: #7c6f9f;
  --viz-reference: #8b8174;
  --viz-grid: color-mix(in srgb, var(--md-sys-color-outline) 35%, transparent);
}

For forecasts specifically:

Meaning	Suggested color direction
User probability	Cyan / primary
Community probability	Deep sea / secondary
Resolved true	Green-teal
Resolved false	Clay-red
Uncertainty interval	Muted violet/gray-blue
Baseline / prior	Stone gray
Stale / inactive	Desaturated neutral

Avoid using red/green alone. Pair color with shape, labels, line style, or iconography.

8. Forecast-card styling

A good forecast card could use this hierarchy:
```
.forecast-card {
  background: var(--md-sys-color-surface-container-low);
  color: var(--md-sys-color-on-surface);
  border: 1px solid var(--md-sys-color-outline-variant);
  border-radius: 24px;
  box-shadow: var(--elevation-1);
}

.forecast-card[data-active="true"] {
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
  border-color: color-mix(in srgb, var(--md-sys-color-primary) 40%, transparent);
}
```
The active card should feel selected, not glowing. Material 3 generally works better with container fills than with thin neon borders.

9. Shape system: classical order + expressive variance

Use a small, deliberate shape scale:

--shape-xs: 6px;     /* small chips, tags */
--shape-sm: 10px;    /* inputs */
--shape-md: 16px;    /* menus, small cards */
--shape-lg: 24px;    /* main cards */
--shape-xl: 32px;    /* hero panels, dialogs */
--shape-pill: 999px;

Recommended usage:

Component	Shape
Buttons	Pill or 16–20px
Forecast cards	20–28px
Dialogs	28–32px
Inputs	10–14px
Chips	Pill
Navigation rail item	Pill
Dense tables	8–12px

The “classical” part should come from layout discipline, not sharp corners. Material 3 Expressive explicitly leans on shape contrast and bolder component expression.

10. Typography

Use a modern UI sans for most text, but introduce a classical accent only in controlled places.

Recommended stack:

--font-ui: Inter, Roboto, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
--font-serif-accent: "Source Serif 4", "Libre Baskerville", Georgia, serif;
--font-mono: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;

Use serif accent for:

landing-page hero headline
section headings
quotes / explanatory philosophy
“oracle”-style empty states, if used carefully

Do not use serif in the app’s operational UI, probability inputs, tables, or dense controls.

11. Texture and material effects

Use texture extremely lightly. Good options:
```
.surface-stone {
  background:
    radial-gradient(circle at 20% 0%, rgb(255 255 255 / 0.35), transparent 35%),
    linear-gradient(180deg, var(--md-sys-color-surface-container-low), var(--md-sys-color-surface));
}
```
For a classical-modern feel:

use very subtle stone noise in landing pages, not dense app views
use bronze gradients only on decorative badges, not buttons
avoid skeuomorphic marble panels
use mosaic motifs as 1–2px geometric dividers or empty-state illustrations
keep high-density screens clean

A forecasting app needs trust. Heavy texture reduces perceived precision.

12. Landing page vs app shell

The landing page can be more thematic. The app should be more utilitarian.

Landing page:

deep sea hero background
cyan CTA
bronze micro-accent
stone cards
optional mosaic/geometric ornament
serif headline

App:

mostly stone/dark-stone surfaces
cyan selected/action states
minimal bronze
charts optimized for readability
serif almost absent

This lets the brand be memorable without compromising daily usability.

13. Component-level rules

Use this decision matrix:

UI element	Preferred styling
Primary CTA	Cyan filled button
Secondary CTA	Sea-blue tonal/outlined button
Dangerous action	Clay-red text/outlined unless very destructive
Forecast probability input	Neutral surface + cyan focus ring
Active forecast	Primary container
Resolved forecast	Semantic status chip, not full-card recolor
Leaderboard / score	Neutral table + bronze/cyan highlights
Calibration warning	Tertiary or error depending severity
App nav	Sea-blue selected state or cyan selected indicator
Empty state	Stone background + line-art/mosaic motif
Modal/dialog	High surface container, large radius, restrained accent
14. Accessibility constraints

Practical rules:

Every token pair like primary/on-primary, surface/on-surface, error/on-error should pass WCAG contrast for body text.
Do not rely on color alone for probability outcomes.
Dark theme charts need independent tuning; simple inversion often fails.
Keep cyan saturation lower in dark mode to avoid visual vibration.
In light mode, pure turquoise can look too low-contrast against white; use darker cyan for text/buttons and pale cyan only for containers.

Material’s role system is specifically designed to pair container colors with readable foreground “on” colors, so lean into that instead of manually picking foregrounds per component.

15. Implementation recommendation

Use a token file structure like:

src/styles/
  tokens/
    color-seeds.css
    color-light.css
    color-dark.css
    shape.css
    typography.css
    elevation.css
    motion.css
    data-viz.css
  theme.css

Then components consume semantic roles only:
```
.button-primary {
  background: var(--md-sys-color-primary);
  color: var(--md-sys-color-on-primary);
}

.panel {
  background: var(--md-sys-color-surface-container);
  color: var(--md-sys-color-on-surface);
}

.badge-insight {
  background: var(--md-sys-color-tertiary-container);
  color: var(--md-sys-color-on-tertiary-container);
}
```
Avoid component-local colors like:
```
/* Avoid */
.card {
  background: #f6f1e7;
}
```
Use:
```
/* Better */
.card {
  background: var(--md-sys-color-surface-container-low);
}
```
16. Recommended visual identity summary

The strongest practical theme would be:

Primary: Aegean cyan
Structure: wine-dark sea blue
Surface: limestone / marble / plaster neutrals
Accent: restrained bronze
Error/warmth: terracotta clay
Style: Material 3 Expressive components with classical proportions and subtle mosaic/stone references

That gives you a distinctive alternative to generic blue SaaS while staying compatible with serious forecasting, data visualization, dark mode, and Material 3 implementati

## Reviewer Two Design Proposals:
CSS Outline:
```
/* Core structure mapping to your classical design language */
:root {
  --font-display: 'Cinzel', serif;
  --font-body: 'Plus Jakarta Sans', sans-serif;
  --radius-mosaic: 24px;

  /* Light Theme Archetype: Travertine Stone & Deep Cyan Sea */
  --md-sys-color-primary: #006A7A;
  --md-sys-color-on-primary: #FFFFFF;
  --md-sys-color-secondary: #3B5461;
  --md-sys-color-tertiary: #7A5900; /* Metallic Accent */
  --md-sys-color-background: #FAF9F6; /* Travertine Warm White */
  --md-sys-color-surface: #FFFDFB;
  --md-sys-color-surface-variant: #DBE4E6; /* Marble Fragment */
  --md-sys-color-on-surface: #191C1D;
}

@media (prefers-color-scheme: dark) {
  :root {
    /* Dark Theme Archetype: Obsidian Night & Electric Turquoise */
    --md-sys-color-primary: #56D6EC;
    --md-sys-color-on-primary: #003640;
    --md-sys-color-secondary: #A3CDDB;
    --md-sys-color-tertiary: #F5BE49; /* Gold Leaf Accent */
    --md-sys-color-background: #111416; /* Volcanic Rock */
    --md-sys-color-surface: #191C1E;
    --md-sys-color-surface-variant: #3F484A;
    --md-sys-color-on-surface: #E1E3E4;
  }
}

/* Application Component Examples */
.dashboard-card {
  background-color: var(--md-sys-color-surface);
  border-radius: var(--radius-mosaic);
  border: 1px solid var(--md-sys-color-surface-variant);
  padding: 24px;
}

.metric-header {
  font-family: var(--font-display);
  color: var(--md-sys-color-primary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
```

## Reviewer Three Design Proposal:
```
Those seven swatch families above are your whole system. A few notes on how I derived them and then the practical rules.
The structure follows Material 3's color roles. M3 doesn't want you picking arbitrary colors — it wants a few key colors (seeds) that each get expanded into a 13-step tonal palette, after which fixed roles pull specific tones. So treat the brand chips above as seeds and let the tool generate the rest. Turquoise is your primary (the hero/brand color), verdigris (oxidized-bronze green) is the secondary that does the quiet supporting work, and bronze/gold is the tertiary — the warm metallic accent that pops against all that cyan. Tyrian purple (the actual murex dye of the ancient Aegean) is best added as an M3 extended/custom color, reserved for rare emphasis rather than thrown into the rotation.
The single most effective move for your theme is in the neutrals. Instead of generating dark surfaces from a plain gray neutral, key the neutral toward a deep desaturated sea-blue. That's what makes the dark theme read as "wine-dark sea" rather than generic charcoal — every elevated surface inherits the tint. In light mode, key the neutral slightly warm so surfaces read as limestone/marble rather than clinical white. Those two decisions carry most of the "classical" feeling without any imagery at all.
Light vs dark, the practical bit. Use the tonal roles, not raw hex, in your CSS — that's what keeps contrast correct automatically. Turquoise at full brightness fails text contrast on white, so on light surfaces your text/icon turquoise should come from the darker tones (around tone 40 and below) and on dark surfaces from the lighter tones (tone 80+). M3's on-primary / on-surface roles already encode this; if you assign colors by role you mostly can't get the contrast wrong. Verify the final pairs hit 4.5:1 for text and 3:1 for UI elements regardless.
Executing "classical meets modern" without it turning kitschy. The trap here is skeuomorphism — literal marble textures, glossy gold-gradient buttons, faux-chiseled bevels. They read as dated, hurt performance, and wreck accessibility. The tasteful version expresses "classical" through restraint and type, not texture:

Type does the heavy lifting. Pair an inscriptional display face with a clean modern sans. Cinzel (free on Google Fonts, literally modeled on Roman inscriptional capitals) is a great wordmark/hero face — used sparingly, all-caps, generous letter-spacing. Then set everything functional (UI, body, every chart label) in a neutral grotesk like Inter or Outfit. Carved Roman caps over a crisp grotesk is your classical-meets-modern tension, done legibly.
Metal and stone live as color and discipline, not material. You already have bronze as a flat accent tone and stone/sea in the neutrals — that's enough. Allow yourself at most one tactile flourish: a very faint stone-grain noise (1–3% opacity) on a hero or empty state, or one thin gold keyline under the wordmark. Reserve any real metallic gradient for the logo alone.
Lean on M3's tonal elevation (surfaces tinted by overlay) rather than heavy drop shadows. The calm, "carved from one block" feel it produces suits a trustworthy forecasting tool better than floating cards.
For M3 Expressive specifically: take the larger, more varied corner radii and the springier motion for navigation and the FAB — but keep chart and data transitions quick and purposeful (~200–300ms, simple easing). A forecasting tool earns trust by feeling precise, not bouncy.

Forecasting-specific color semantics. This is the part a pure brand palette usually misses. Your categorical series palette (bottom row above) is built so the two colors you'll use most — turquoise for actuals, bronze/gold for forecast — are far apart in hue and stay distinguishable for the common colorblindness types. For confidence/prediction intervals, fill with translucent turquoise (your primary at low alpha) so the band reads as "the same series, less certain." For anything diverging (over/under forecast, up/down surprise), go teal-to-terracotta rather than the usual red/green, which collapses for red-green colorblind users. And as a rule, never encode forecast meaning by hue alone — pair it with a dash pattern, marker, or direct label (actuals solid, forecast dashed is the classic, robust choice).
Implementation and PWA. Generate the full token set once from your key colors using the Material Theme Builder (or material-color-utilities in code), export it as CSS custom properties, and ship it as a documented tokens.css. Since you're open source, add a one-line norm to CONTRIBUTING — "use the role tokens, never raw hex" — so contributors don't reintroduce ad-hoc colors. Drive theming with prefers-color-scheme plus a manual toggle persisted in localStorage, switching via a data-theme attribute on the root. For the PWA chrome: set theme_color and background_color in the manifest, and use <meta name="theme-color"> with light/dark media queries so the browser bar and status bar match each theme. I'd skip Material You dynamic color for a branded app — it dilutes the identity you're deliberately building; offer it at most as an optional "match my system" setting.
One judgment call worth making early: decide whether bronze/gold or Tyrian purple is your "second voice." I'd default to bronze as tertiary (warmer, more legible, very Roman-mosaic) and keep purple as the rare extended accent — but if you want the app to feel more regal than coastal, you can swap their roles. Want me to take a pass at the actual tokens.css with both light and dark role assignments filled in, so you have something to drop in directly?
```
## Shapes
Greco-Roman themes can be incorporated in shapes, not so much by including symbols or drawings which can look kitsch, but by how the items are laid out and arranged in the app. 
Mosaic tiles can influence the design of any markers on the line graphs.

## Making metallic a proper metal look. 
Realistic metal is mostly about highlights, anisotropic streaks, and abrupt tonal shifts, not just hue. Realistic metal wants high dynamic range and directional highlight behavior. However, it is very hard to do well in an app design.
If it is a symbol/object/badge: metallic interior, flat MD3 border.
If it is a core control: no metallic treatment.

Generally true metallic look should be used consistently but relatively rarely, on key symbols or "ceremonial" objects.

Examples:
```
.metal-object {
  --metal-angle: 220deg;

  background:
    conic-gradient(
      from var(--metal-angle),
      var(--metal-bronze-dark) 0deg,
      var(--metal-bronze-mid) 42deg,
      var(--metal-bronze-light) 78deg,
      var(--metal-bronze-mid) 112deg,
      var(--metal-bronze-dark) 156deg,
      var(--metal-bronze-mid) 214deg,
      var(--metal-bronze-light) 252deg,
      var(--metal-bronze-mid) 292deg,
      var(--metal-bronze-dark) 360deg
    );

  border: 1px solid var(--md-sys-color-outline-variant);
  box-shadow: var(--md-sys-elevation-1);
}
```
on hover:
```
.metal-object {
  --metal-angle: 220deg;

  background:
    conic-gradient(
      from var(--metal-angle),
      var(--metal-bronze-dark) 0deg,
      var(--metal-bronze-mid) 42deg,
      var(--metal-bronze-light) 78deg,
      var(--metal-bronze-mid) 112deg,
      var(--metal-bronze-dark) 156deg,
      var(--metal-bronze-mid) 214deg,
      var(--metal-bronze-light) 252deg,
      var(--metal-bronze-mid) 292deg,
      var(--metal-bronze-dark) 360deg
    );

  border: 1px solid var(--md-sys-color-outline-variant);
  box-shadow: var(--md-sys-elevation-1);
}
```