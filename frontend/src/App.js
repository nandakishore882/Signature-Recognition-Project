import { useState, useRef } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --ink: #0a0a0f;
    --paper: #f5f2eb;
    --accent: #c8392b;
    --accent2: #2b6cc8;
    --muted: #8a8680;
    --border: #d4cfc6;
    --success: #1a7a4a;
    --card: #ffffff;
  }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
  }

  .app {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  .header {
    margin-bottom: 56px;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 24px;
  }

  .header-tag {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .header h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
  }

  .header h1 span { color: var(--accent); }

  .header-sub {
    margin-top: 12px;
    font-size: 13px;
    color: var(--muted);
    max-width: 480px;
    line-height: 1.6;
  }

  .steps-bar {
    display: flex;
    gap: 0;
    margin-bottom: 32px;
    border: 1.5px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .step {
    flex: 1;
    padding: 10px 14px;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
    border-right: 1.5px solid var(--border);
    transition: all 0.2s;
  }

  .step:last-child { border-right: none; }

  .step.done {
    color: var(--success);
    background: #f0faf4;
  }

  .step.active {
    color: var(--ink);
    background: #fffef9;
    font-weight: 500;
  }

  .step-num {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 1.5px solid currentColor;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
  }

  .step.done .step-num {
    background: var(--success);
    border-color: var(--success);
    color: white;
  }

  .upload-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }

  @media (max-width: 600px) {
    .upload-grid { grid-template-columns: 1fr; }
    .step-text { display: none; }
  }

  .upload-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .upload-card:hover { border-color: var(--ink); }
  .upload-card.active { border-color: var(--accent2); }

  .card-label {
    padding: 14px 18px;
    border-bottom: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .card-num {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 18px;
    color: var(--accent);
  }

  .card-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 14px;
  }

  .card-desc {
    font-size: 11px;
    color: var(--muted);
    margin-left: auto;
    letter-spacing: 0.05em;
  }

  .drop-zone {
    padding: 32px 18px;
    text-align: center;
    cursor: pointer;
    transition: background 0.15s;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
  }

  .drop-zone:hover { background: #f9f7f3; }

  .drop-icon {
    width: 44px;
    height: 44px;
    border: 1.5px dashed var(--border);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: var(--muted);
    transition: all 0.2s;
  }

  .drop-zone:hover .drop-icon {
    border-color: var(--ink);
    color: var(--ink);
  }

  .drop-text {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
  }

  .drop-text strong {
    color: var(--ink);
    display: block;
    font-size: 13px;
    margin-bottom: 2px;
  }

  .preview-img {
    width: 100%;
    max-height: 160px;
    object-fit: contain;
    padding: 12px;
    background: #fafaf8;
  }

  .preview-footer {
    padding: 8px 18px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .preview-name {
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 70%;
  }

  .preview-name span { color: var(--success); }

  .change-btn {
    font-size: 11px;
    color: var(--accent2);
    background: none;
    border: none;
    cursor: pointer;
    font-family: 'DM Mono', monospace;
    padding: 2px 6px;
    border-radius: 3px;
    transition: background 0.15s;
    flex-shrink: 0;
  }

  .change-btn:hover { background: #eef3fc; }

  .action-row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    margin-bottom: 8px;
  }

  .verify-btn {
    padding: 16px;
    background: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
  }

  .verify-btn:hover:not(:disabled) {
    background: var(--accent);
    transform: translateY(-1px);
  }

  .verify-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .verify-btn.loading {
    background: var(--muted);
    cursor: wait;
  }

  .reset-btn {
    padding: 16px 20px;
    background: #fff0ee;
    border: 1.5px solid #f5c6c0;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    color: var(--accent);
    transition: all 0.2s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .reset-btn:hover {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  .hint {
    font-size: 11px;
    color: var(--muted);
    text-align: center;
    margin-bottom: 24px;
  }

  .result-card {
    margin-top: 32px;
    border: 2px solid var(--ink);
    border-radius: 4px;
    overflow: hidden;
    animation: slideUp 0.3s ease;
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-header {
    padding: 16px 24px;
    background: var(--ink);
    color: var(--paper);
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .result-header-reset {
    background: none;
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 5px 12px;
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .result-header-reset:hover { background: rgba(255,255,255,0.15); }

  .result-body {
    padding: 28px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }

  .result-verdict {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(36px, 6vw, 56px);
    letter-spacing: -0.03em;
    line-height: 1;
  }

  .result-verdict.real { color: var(--success); }
  .result-verdict.forged { color: var(--accent); }

  .result-meta {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
    min-width: 200px;
  }

  .result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }

  .result-row:last-child { border-bottom: none; }
  .result-row-label { color: var(--muted); letter-spacing: 0.05em; }

  .result-row-value {
    font-weight: 500;
    font-family: 'Syne', sans-serif;
  }

  .confidence-bar {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 4px;
    overflow: hidden;
  }

  .confidence-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
  }

  .confidence-fill.real { background: var(--success); }
  .confidence-fill.forged { background: var(--accent); }

  .try-again-banner {
    padding: 14px 24px;
    background: #f9f7f3;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .try-again-text {
    font-size: 12px;
    color: var(--muted);
  }

  .try-again-text strong { color: var(--ink); }

  .try-again-big-btn {
    padding: 10px 24px;
    background: var(--ink);
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .try-again-big-btn:hover { background: var(--accent2); }

  .error-msg {
    margin-top: 16px;
    padding: 14px 18px;
    background: #fff0ee;
    border: 1px solid #f5c6c0;
    border-radius: 4px;
    font-size: 13px;
    color: var(--accent);
  }

  /* Contributors */
  .contributors {
    margin-top: 48px;
    padding-top: 32px;
    border-top: 2px solid var(--ink);
  }

  .contributors-label {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
  }

  .contributors-grid {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .contributor-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 4px;
    text-decoration: none;
    color: var(--ink);
    transition: all 0.2s;
    flex: 1;
    min-width: 200px;
  }

  .contributor-card:hover {
    border-color: var(--ink);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }

  .contributor-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--ink);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--paper);
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 14px;
    flex-shrink: 0;
    overflow: hidden;
  }

  .contributor-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .contributor-info { flex: 1; min-width: 0; }

  .contributor-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .contributor-handle {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }

  .contributor-gh {
    font-size: 16px;
    color: var(--muted);
    transition: color 0.2s;
    flex-shrink: 0;
  }

  .contributor-card:hover .contributor-gh { color: var(--ink); }

  .footer {
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
`;

const CONTRIBUTORS = [
  {
    name: "Nanda Kishore",
    handle: "@nandakishore882",
    github: "https://github.com/nandakishore882",
    linkedin: "https://www.linkedin.com/in/nandakishore882/",
    initials: "NK",
  },
  {
    name: "Anil Kumar",
    handle: "@coming-soon",
    github: "#",
    linkedin: "#",
    initials: "AK",
  },
  {
    name: "Meher Anand",
    handle: "@coming-soon",
    github: "#",
    linkedin: "#",
    initials: "MA",
  },
];

export default function App() {
  const [refImage, setRefImage] = useState(null);
  const [testImage, setTestImage] = useState(null);
  const [refPreview, setRefPreview] = useState(null);
  const [testPreview, setTestPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refInput = useRef();
  const testInput = useRef();

  const handleFile = (file, type) => {
    if (!file) return;
    const preview = URL.createObjectURL(file);
    if (type === 'ref') { setRefImage(file); setRefPreview(preview); }
    else { setTestImage(file); setTestPreview(preview); }
    setResult(null);
    setError(null);
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, type);
  };

  const verify = async () => {
    if (!refImage || !testImage) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('reference', refImage);
      formData.append('test', testImage);

      const res = await fetch('https://signature-recognition-api.onrender.com/verify', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      if (data.error) setError(data.error);
      else setResult(data);
    } catch (err) {
      setError('Could not connect to the server. Make sure Flask is running on port 5000. Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setRefImage(null); setTestImage(null);
    setRefPreview(null); setTestPreview(null);
    setResult(null); setError(null);
  };

  // FIX 1: Step logic based on individual upload state, not combined
  const refDone = !!refImage;
  const testDone = !!testImage;
  const bothDone = refDone && testDone;
  const resultDone = !!result;

  const getStepState = (n) => {
    if (n === 1) return refDone ? 'done' : 'active';
    if (n === 2) return testDone ? 'done' : refDone ? 'active' : '';
    if (n === 3) return resultDone ? 'done' : bothDone ? 'active' : '';
    if (n === 4) return resultDone ? 'done' : '';  // FIX 2: Result step ticks when result exists
    return '';
  };

  const UploadCard = ({ type, image, preview, inputRef, num, title, desc }) => (
    <div className={`upload-card ${preview ? 'active' : ''}`}>
      <div className="card-label">
        <span className="card-num">{num}</span>
        <span className="card-title">{title}</span>
        <span className="card-desc">{desc}</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => handleFile(e.target.files[0], type)}
      />
      {preview ? (
        <>
          <img src={preview} alt="preview" className="preview-img" />
          <div className="preview-footer">
            <div className="preview-name">
              <span>✓</span> {image?.name}
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button className="change-btn" onClick={() => inputRef.current.click()}>
                Change
              </button>
              <button className="change-btn" style={{ color: 'var(--accent)' }}
                onClick={() => {
                  if (type === 'ref') { setRefImage(null); setRefPreview(null); }
                  else { setTestImage(null); setTestPreview(null); }
                  setResult(null); setError(null);
                }}>
                Remove
              </button>
            </div>
          </div>
        </>
      ) : (
        <div
          className="drop-zone"
          onClick={() => inputRef.current.click()}
          onDrop={e => handleDrop(e, type)}
          onDragOver={e => e.preventDefault()}
        >
          <div className="drop-icon">↑</div>
          <div className="drop-text">
            <strong>Click or drag to upload</strong>
            PNG, JPG, JPEG supported
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        <div className="header">
          <div className="header-tag">Deep Learning · Signature Verification</div>
          <h1>Signature<br /><span>Verification</span></h1>
          <p className="header-sub">
            Upload a genuine reference signature and a test signature. Our model will compare them and determine authenticity.
          </p>
        </div>

        {/* Step indicator - FIX: each step is independent */}
        <div className="steps-bar">
          {[
            { n: 1, label: 'Upload Reference' },
            { n: 2, label: 'Upload Test' },
            { n: 3, label: 'Verify' },
            { n: 4, label: 'Result' },
          ].map(s => (
            <div key={s.n} className={`step ${getStepState(s.n)}`}>
              <div className="step-num">
                {getStepState(s.n) === 'done' ? '✓' : s.n}
              </div>
              <span className="step-text">{s.label}</span>
            </div>
          ))}
        </div>

        <div className="upload-grid">
          <UploadCard
            type="ref" image={refImage} preview={refPreview} inputRef={refInput}
            num="01" title="Reference" desc="GENUINE"
          />
          <UploadCard
            type="test" image={testImage} preview={testPreview} inputRef={testInput}
            num="02" title="Test Signature" desc="TO VERIFY"
          />
        </div>

        <div className="action-row">
          <button
            className={`verify-btn ${loading ? 'loading' : ''}`}
            onClick={verify}
            disabled={!refImage || !testImage || loading}
          >
            {loading ? '[ Analyzing... ]' : '[ Verify Signature ]'}
          </button>
          <button className="reset-btn" onClick={reset} title="Start over">
            ↺ Reset
          </button>
        </div>

        {!refImage && !testImage && (
          <p className="hint">↑ Upload both signatures above, then click Verify</p>
        )}

        {error && <div className="error-msg">⚠ {error}</div>}

        {result && (
          <div className="result-card">
            <div className="result-header">
              <span>Analysis Result</span>
              <button className="result-header-reset" onClick={reset}>
                ↺ New Verification
              </button>
            </div>
            <div className="result-body">
              <div className={`result-verdict ${result.result.toLowerCase()}`}>
                {result.result}
              </div>
              <div className="result-meta">
                <div className="result-row">
                  <span className="result-row-label">VERDICT</span>
                  <span className="result-row-value">
                    {result.result === 'Real' ? '✓ Genuine Signature' : '✗ Forged Signature'}
                  </span>
                </div>
                <div className="result-row">
                  <span className="result-row-label">CONFIDENCE</span>
                  <span className="result-row-value">{result.confidence}%</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className={`confidence-fill ${result.result.toLowerCase()}`}
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
                <div className="result-row">
                  <span className="result-row-label">SIMILARITY SCORE</span>
                  <span className="result-row-value">{result.similarity}</span>
                </div>
              </div>
            </div>
            <div className="try-again-banner">
              <div className="try-again-text">
                <strong>Want to verify another signature?</strong><br />
                Click the button to start a new verification.
              </div>
              <button className="try-again-big-btn" onClick={reset}>
                ↺ Start New Verification
              </button>
            </div>
          </div>
        )}

        {/* Contributors Section */}
        <div className="contributors">
          <div className="contributors-label">Built by</div>
          <div className="contributors-grid">
            {CONTRIBUTORS.map((c) => (
              <div key={c.name} className="contributor-card">
                <div className="contributor-avatar">
                  {c.github !== '#' ? (
                    <img
                      src={`https://github.com/${c.handle.replace('@', '')}.png?size=72`}
                      alt={c.name}
                      onError={e => { e.target.style.display = 'none'; }}
                    />
                  ) : null}
                  {c.initials}
                </div>
                <div className="contributor-info">
                  <div className="contributor-name">{c.name}</div>
                  <div className="contributor-handle">{c.handle}</div>
                </div>
                <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                  {c.github !== '#' ? (
                    <a href={c.github} target="_blank" rel="noreferrer"
                      style={{ fontSize: '13px', color: 'var(--muted)', textDecoration: 'none', border: '1px solid var(--border)', padding: '3px 8px', borderRadius: '3px' }}
                      title="GitHub">GH</a>
                  ) : null}
                  {c.linkedin !== '#' ? (
                    <a href={c.linkedin} target="_blank" rel="noreferrer"
                      style={{ fontSize: '13px', color: 'var(--accent2)', textDecoration: 'none', border: '1px solid var(--border)', padding: '3px 8px', borderRadius: '3px' }}
                      title="LinkedIn">in</a>
                  ) : null}
                  {c.github === '#' && (
                    <span style={{ fontSize: '11px', color: 'var(--muted)' }}>soon</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="footer">
          <span>Signature Recognition · Siamese Neural Network</span>
          <span>Model: CNN + Siamese · Input: 32×32px</span>
        </div>
      </div>
    </>
  );
}