# Level 4 Implementation Plan

## Art Direction

- [ ] Decide what "Indian inspired" means for this game before drawing: Indigenous North American-inspired, South Asian-inspired, or another specific culture.
- [ ] Avoid caricature and sacred/ceremonial designs unless there is a clear reason and proper reference. Prefer original fantasy-adventure costumes inspired by materials, silhouettes, colors, and patterns.
- [ ] Create a small style sheet: character height, outline weight, palette, platform style, arrow/axe silhouettes, plume look, and blood-effect limits.
- [ ] Use placeholder sprites first, then replace them with final sprites once the mechanics feel correct.

## Where To Design Or Find Artwork

- [ ] Design custom sprites in [Aseprite](https://www.aseprite.com/) if paid tooling is fine. It supports animation frames, tags, sprite-sheet export, and pixel-art tooling.
- [ ] Design custom sprites in [Piskel](https://www.piskelapp.com/) if you want a free browser/desktop workflow with animated sprite export.
- [ ] Use [Lospec palettes](https://lospec.com/palettes/) to pick a controlled color palette before drawing the characters and effects.
- [ ] Search [itch.io 2D sprite assets](https://itch.io/game-assets/tag-2d/tag-sprites) for paid/free sprite sheets, especially as temporary reference or placeholders.
- [ ] Search [OpenGameArt character sprites](https://opengameart.org/content/character-sprites) for free/open assets, but check each asset license and attribution requirement.
- [ ] Consider [Kenney Game Assets All-in-1](https://kenney.itch.io/kenney-game-assets) for broad placeholder packs, icons, UI pieces, and effects.
- [ ] Store final art under `images/level_4/` with consistent names such as `player_body.png`, `player_parachute.png`, `arrow.png`, `axe_slash.png`, `plume.png`, and `blood_particles.png`.
- [ ] Add an `images/level_4/credits.md` file for asset licenses, author names, links, and any attribution text.

## Artwork Motion

- [ ] Add idle animation: subtle breathing, small head/shoulder bob, and a gentle weapon sway.
- [ ] Add parachute animation: canopy open pop, rope tension, and a slow cloth wobble while gliding.
- [ ] Add heat interaction animation: plume and parachute flutter more when `heat_lift > 0`.
- [ ] Add attack animation states: bow draw/release and axe wind-up/slash/recover.
- [ ] Build a lightweight animation helper that supports frame lists, frame duration, looping/non-looping clips, and per-player tinting.

## Player Combat

- [ ] Add player state fields: `health`, `facing`, `attack_cooldown`, `invuln_timer`, `weapon_mode`, and `plume_level`.
- [ ] Map phone controls for Level 4: `A` or `UP` jump, `B` toggle parachute, `DOWN` switch/hold weapon mode or fast-fall, `START` reset.
- [ ] Decide attack input: either tap `A` while parachute is open for bow, or add `SELECT` as attack if the phone UI is expanded.
- [ ] Implement arrows as projectile objects with position, velocity, owner id, lifetime, damage, and collision rect.
- [ ] Implement axe slash as a short-lived arc/hitbox in front of the character with knockback and cooldown.
- [ ] Prevent self-hits and repeated hits from the same slash instance.
- [ ] Add a small debug overlay for projectile count, hitboxes, and player health.

## Plume Reward

- [ ] When a player hits an opponent with an arrow, increment the attacker plume level.
- [ ] Draw plume level as a visible feather/plume stack or aura behind the character.
- [ ] Make plume reward gameplay-relevant but not overpowering: possible options are slightly stronger heat lift, wider parachute heat probe, or a temporary movement boost.
- [ ] Add a cap and decay rule, for example max `3` plume stacks and slow decay after `10` seconds without hits.
- [ ] Add hit feedback: plume burst, short sparkle, or warm glow on the attacker.

## Blood Animation

- [ ] Keep blood stylized and readable: small red pixel particles or short splash arcs, not realistic gore.
- [ ] Spawn particles on arrow/axe hit with position, velocity, gravity, lifetime, and fade alpha.
- [ ] Add a low-particle mode if performance drops during camera processing.
- [ ] Make blood effects optional with a constant such as `SHOW_BLOOD_EFFECTS = True`.
- [ ] Ensure blood particles are visual-only and do not enter shadow/heat detection masks.

## Integration Order

- [ ] Step 1: Add asset folders, `credits.md`, and placeholder sprite loading.
- [ ] Step 2: Replace procedural player drawing with sprite-based drawing while keeping old collision boxes.
- [ ] Step 3: Add idle/parachute/heat animations.
- [ ] Step 4: Add arrows and projectile collision.
- [ ] Step 5: Add axe slash hitboxes.
- [ ] Step 6: Add plume reward and display.
- [ ] Step 7: Add blood particles and a toggle constant.
- [ ] Step 8: Playtest with two phones and tune cooldowns, lift, projectile speed, and readability on the projector.

## Open Questions

- [ ] Which interpretation of "Indian inspired" should the game use?
- [ ] Should the game be non-lethal/stylized, or is visible blood definitely part of the design?
- [ ] Should each phone choose bow vs axe, or should every character have both?
- [ ] Should plume be cosmetic only, or should it affect gameplay?
