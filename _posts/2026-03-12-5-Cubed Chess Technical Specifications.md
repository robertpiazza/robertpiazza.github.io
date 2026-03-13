# 5³ Chess (Raumschach) Technical Blog

## Overview

5³ Chess is a browser-based 3-D chess game played on a 5×5×5 board (Raumschach variant). It runs entirely client-side using Three.js for 3-D rendering and Firebase Realtime Database for optional two-player online play. No server-side compute is required beyond static file hosting.

**[Play now](https://robertpiazza.com/5-3-Chess/)**

---

## Technology Stack

| Layer | Technology |
|---|---|
| Rendering | Three.js 0.165 (ESM, CDN) |
| Multiplayer sync | Firebase Realtime Database 10.12.2 |
| Language | Vanilla JavaScript (ES Modules) |
| Styling | Plain CSS |
| Build tooling | None — direct browser execution via importmap |

---

## Board & Coordinate System

The board is a 5×5×5 grid. Each cell is addressed as `(x, y, z)` where all axes range from 0–4.

- **x**: left–right (file)
- **y**: front–back (rank)
- **z**: bottom–top (layer), with White starting at z=0–1 and Black at z=3–4

### View Modes

Two view modes are supported and can be toggled at any time during play via the hamburger menu:

**Layers view** (default): Five horizontal 5×5 boards stacked vertically, separated by `LAYER_GAP = 1.65` world units. Each square is a flat `BoxGeometry` tile.

**Cube view**: All 125 cells rendered as a single 5×5×5 transparent cube lattice with wireframe edges. Pieces appear floating inside the cube at their correct 3-D positions.

Both views share the same coordinate system and move validation logic.

---

## Piece Types

| Piece | Symbol | Movement |
|---|---|---|
| Rook | R | Slides any distance along one axis — X, Y, or Z (6 directions) |
| Bishop | B | Slides along face diagonals — exactly 2 axes change simultaneously (12 directions) |
| Unicorn | U | Slides along space diagonals — all 3 axes change simultaneously (8 directions) |
| Queen | Q | Combines Rook + Bishop + Unicorn — all 26 sliding directions |
| King | K | One step in any of the 26 directions; must stay out of check |
| Knight | N | L-shape in 3-D: 2 squares on one axis + 1 on another; jumps over pieces (up to 24 targets) |
| Pawn | P | Advances one layer (White: +z, Black: −z); captures diagonally forward; promotes to Queen on far layer |

---

Raumschach adds the unicorn piece not found in standard chess.

## 3-D Piece Models

Pieces are loaded as GLB models from the `/web/models/` directory. Each piece type has a corresponding `.glb` file (`king.glb`, `queen.glb`, etc.).

### Proportional Height System

Piece heights are based on real-world standard chess piece proportions, anchored so that the King reaches `TARGET_HEIGHT = 1.50` world units.

| Piece | Real-world height (cm) | In-game height (world units) |
|---|---|---|
| King | 9.5 | 1.50 |
| Queen | 8.5 | 1.342 |
| Bishop | 7.0 | 1.105 |
| Knight | 6.0 | 0.947 |
| Unicorn | 6.0 | 0.947 |
| Rook | 5.5 | 0.868 |
| Pawn | 5.0 | 0.789 |

The Unicorn shares the Knight's body height (excluding the horn). The horn geometry is scaled by the same `knight/9.5` ratio relative to the king so that the overall visual proportions remain consistent.

Each model is normalized at load time by computing its bounding box, scaling uniformly so the tallest axis reaches the target height, then repositioning so the base sits at y=0.

In cube view, all pieces are scaled by an additional `0.83×` factor (`board.pieceScale`) to fit comfortably within the lattice cells.

---

## Game Modes

### Local Play
Two players share the same device and take turns clicking. No network connection required.

### vs AI
One player chooses a color (White, Black, or Random). The AI controls the other color. After the human makes a move, the AI responds automatically.

### Online Multiplayer (Create / Join)
One player creates a game and receives a 6-character room code. The other player enters the code to join. All moves are synchronized in real time via Firebase Realtime Database at `games/{code}/moves`.

---

## AI Opponent

The AI uses **minimax search with alpha-beta pruning**.

### Algorithm

```
findBestMove(board, aiColor, depth = 3)
  → iterates all legal moves for aiColor
  → applies each move to a cloned board
  → calls minimax(depth - 1, opponentTurn, α, β)
  → returns the move with the highest score
```

Terminal conditions:
- No legal moves + in check → checkmate (score ±999999, scaled by depth to prefer faster mates)
- No legal moves + not in check → stalemate (score 0)
- depth == 0 → static material evaluation

### Material Values

| Piece | Value |
|---|---|
| Pawn | 100 |
| Knight | 350 |
| Bishop | 350 |
| Unicorn | 475 |
| Rook | 525 |
| Queen | 1000 |
| King | 20000 |

Evaluation is purely material (no positional tables). Positive scores favor the AI; negative scores favor the opponent.

Pawn promotion to Queen is simulated inside the search tree.

---

## Hint System

A hint button (💡) is displayed in the bottom-right corner of the screen. It is enabled during the human player's turn in both Local Play and vs AI modes.

### Behavior

1. Player clicks 💡. The button enters a pulsing CSS animation (`hint-pulse`) to signal computation in progress.
2. After a 50 ms delay (allowing the animation frame to render), `findBestMove` is called at depth 2 for the current player's color.
3. The suggested move is visualized using two colored overlays on the board without executing the move:
   - **Cyan** — the piece to move (source cell)
   - **Orange** — the destination cell
4. Hint highlights are stored in a separate `hintMeshes` array, independent of the selection highlight array, so they persist until the player makes their own move or requests a new hint.
5. Hint highlights are cleared automatically when any move is executed or when the view mode is toggled.

In ring geometry (layers view), hints use `THREE.RingGeometry`. In cube view, hints use `THREE.LineSegments` edge boxes.

---

## Undo Move

An **Undo** button is available in the hamburger menu. Its behavior depends on the game mode.

### Local Play and vs AI

- The game state maintains a history stack (`_history`). Before each move is applied, a snapshot of the full board state is pushed: `{ board, currentTurn, status, moveCount }`.
- Clicking Undo pops the latest snapshot and restores the game state immediately.
- In vs AI mode, undoing once steps back before the AI's last move, restoring the board to the state before the human's prior move (one human ply + one AI ply are both undone in a single undo operation by popping twice).

### Online Multiplayer

Network undo uses a **request/approval protocol** over Firebase.

1. The requesting player clicks Undo. A push event `{ type: 'request', senderColor }` is written to `games/{code}/undoEvents`.
2. The UI shows a "Undo pending…" message to the requester and disables further input.
3. The opponent sees an overlay: "**[Color] requests undo — Accept / Decline**"
4. If the opponent clicks **Accept**, a push event `{ type: 'approved', senderColor }` is written. Both clients receive it; the requesting client calls `undoMove()` and syncs the result.
5. If the opponent clicks **Decline**, a push event `{ type: 'declined', senderColor }` is written. The requester sees "Undo declined" and play resumes.

Seen-key deduplication (`_seenUndoKeys` Set) prevents each client from processing its own echo of the event it sent.

---

## Network Protocol (Firebase)

All real-time game data is stored under `games/{roomCode}/` in Firebase Realtime Database.

| Path | Description |
|---|---|
| `games/{code}/moves` | Append-only list of moves in `{src, dst}` format |
| `games/{code}/undoEvents` | Append-only list of undo request/response events |

Moves are written with `push()` and read with `onValue()` / `onChildAdded()`. The host player is always White; the joining player is always Black.

---

## UI / HUD

### HUD (top-left)
- Game title: **5³ Chess**
- Current turn indicator (highlights White or Black)
- Player color indicator (network mode only)
- Status display (check, checkmate, stalemate messages)

### Hamburger Menu (top-right)
A collapsible dropdown triggered by a three-line button. Items:
- **New Game** — returns to lobby
- **Cube View / Layer View** — toggles board rendering mode
- **Undo** — requests undo (disabled when not applicable)
- **? How pieces move** — opens the piece reference modal

The dropdown closes automatically when clicking outside it, or after any button action.

### Hint Button (bottom-right)
Fixed circular button (48 px diameter) displaying 💡. Disabled during opponent's turn and AI thinking. Pulses cyan while the AI calculates the suggestion.

### Piece Reference Modal
An overlay table listing all seven piece types with their movement descriptions. Accessible from the hamburger menu at any time.

### Lobby
The start screen presented before a game begins. Panels:
- **Mode selection**: Create Game, Join Game, Local Play, vs AI
- **Host panel**: displays the shareable 6-character room code
- **Join panel**: text input for entering a room code
- **vs AI panel**: color selection (White / Black / Random)

### Game-Over Overlay
Displayed on checkmate or stalemate. Shows result text and a **Play Again** button that returns to the lobby.

### Undo Request Overlay
A bottom-center toast shown to the opponent when an undo is requested in network play. Contains the request message and Accept / Decline buttons.

---

## File Structure

```
web/
├── index.html              # App shell, overlay markup, hamburger JS
├── css/
│   └── style.css           # All styles (HUD, overlays, animations)
└── js/
    ├── main.js             # Entry point: scene setup, game orchestration
    ├── gameState.js        # Board state, move execution, undo history
    ├── board.js            # Three.js board geometry (layers & cube views)
    ├── pieces.js           # GLB model loading, normalization, placement
    ├── moveValidator.js    # Legal move generation, check detection
    ├── ai.js               # Minimax + alpha-beta, findBestMove()
    ├── inputHandler.js     # Raycasting, click handling, move execution flow
    ├── network.js          # Firebase adapter (moves + undo events)
    └── ui.js               # DOM helpers for overlays and buttons
models/
├── king.glb
├── queen.glb
├── bishop.glb
├── knight.glb
├── unicorn.glb
├── rook.glb
└── pawn.glb
```

---

## Development & Local Serving

Because the project uses ES Modules loaded from CDN, it must be served over HTTP (not opened as a `file://` URL). Any static file server works:

```bash
# Python (built-in)
python3 -m http.server 8080 --directory web

# Node (npx, no install needed)
npx serve web

# VS Code: Live Server extension → right-click index.html → "Open with Live Server"
```

Then open `http://localhost:8080` in a browser.

Firebase credentials are embedded in `main.js`. For offline-only use (Local Play and vs AI), Firebase connectivity is not required; only network features will be unavailable.
