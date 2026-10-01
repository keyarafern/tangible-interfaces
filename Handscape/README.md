# Living Wall Sketches

A p5.js sketch that displays your webcam feed full-screen, mirrored like a selfie camera, with MediaPipe hand tracking that follows your index fingertip.

## Requirements

- A modern web browser (Chrome, Firefox, Safari, or Edge)
- Python 3 (already installed on macOS) or any simple static file server
- A webcam and permission to use it when the browser prompts you

## Quick start

1. Open a terminal and go to this project folder:

   ```bash
   cd Handscape
   ```

2. Start a local web server:

   ```bash
   python3 -m http.server 8000
   ```

   Leave this terminal window open while you work.

3. Open the sketch in your browser:

   ```
   http://localhost:8000
   ```

4. When prompted, allow camera access. You should see your mirrored webcam feed filling the window.

5. Open the browser developer console (Chrome/Firefox: `Option+Command+J`, Safari: enable the Develop menu first, then **Develop → Show JavaScript Console**). Hold your index finger up to the camera — a red circle should track your fingertip, and coordinates will log to the console like:

   ```
   Index fingertip: x=512.3, y=284.7
   ```

6. To stop the server, press `Ctrl+C` in the terminal.

## Why a local server?

Browsers block webcam access on `file://` URLs. Serving the page over `http://localhost` lets the camera work safely.

## Project files

| File        | Purpose                                      |
|-------------|----------------------------------------------|
| `index.html`| Loads p5.js from a CDN and `sketch.js`       |
| `sketch.js` | Webcam capture, MediaPipe hand tracking, fingertip overlay |

## Troubleshooting

- **No camera prompt / black screen:** Check that another app is not using the webcam, then reload the page and allow camera access.
- **Port already in use:** Pick another port, e.g. `python3 -m http.server 8080`, and open `http://localhost:8080` instead.
