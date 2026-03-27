import React, { useState, useRef, useEffect } from "react";

type ThresholdMode = "fixed" | "otsu" | "percentile";

type SegImage = {
  png_b64: string;
};

type SegmentationPayload = {
  maps: {
    mask: SegImage;
    overlay: SegImage;
    prob: SegImage;
    distance: SegImage;
    contour: SegImage;
  };
  coverage: number;
  thr_used: number;
  num_components: number;
};

type PredictResponse = {
  non_biopsy: boolean;
  reason: string;
  pred_class?: number;
  pred_class_name_ru?: string;
  confidence?: number;
  segmentation?: SegmentationPayload | null;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: string;
  fileName?: string;
  results?: PredictResponse;
  status?: "loading" | "error" | "done";
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<PredictResponse | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const userMsgId = Date.now().toString();
    const previewUrl = URL.createObjectURL(file);
    
    setMessages(prev => [...prev, {
      id: userMsgId,
      role: "user",
      content: `Analyze this biopsy sample: ${file.name}`,
      image: previewUrl,
      fileName: file.name
    }]);

    const assistantMsgId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: assistantMsgId,
      role: "assistant",
      content: "Analyzing sample... (AI Sonnet 4.6)",
      status: "loading"
    }]);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("threshold_mode", "otsu");
      
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed");

      setMessages(prev => prev.map(m => m.id === assistantMsgId ? {
        ...m,
        content: `I've finished the analysis for **${file.name}**. I've opened the results in the artifact panel on the right.`,
        status: "done",
        results: data
      } : m));

      setSelectedArtifact(data);

    } catch (e) {
      setMessages(prev => prev.map(m => m.id === assistantMsgId ? {
        ...m,
        content: `Error: ${e instanceof Error ? e.message : "Analysis failed"}`,
        status: "error"
      } : m));
    }
  };

  return (
    <div className="claude-container">
      {/* Sidebar */}
      <aside className={`claude-sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <div className="claude-logo-text">Biopsy AI</div>
          <button className="new-chat-btn" onClick={() => { setMessages([]); setSelectedArtifact(null); }}>
             <span>+</span> New Analysis
          </button>
        </div>
        <div className="sidebar-content">
          <div className="history-label">Recents</div>
          <div className="history-item active">Current Biopsy</div>
          <div className="history-item">Gastric #01</div>
          <div className="history-item">Intestinal #04</div>
        </div>
        <div className="sidebar-footer">
           <div className="user-profile">
              <div className="user-avatar">IA</div>
              <div className="user-info">
                 <span className="user-name">Ibrohim Avazov</span>
                 <span className="user-plan">Claude Code Base</span>
              </div>
           </div>
        </div>
      </aside>

      {/* Main View Area */}
      <main className={`claude-main ${selectedArtifact ? "artifact-active" : ""}`}>
        
        {/* Chat Section */}
        <section className="chat-section">
           <header className="main-header">
              <button onClick={() => setSidebarOpen(!sidebarOpen)} className="sidebar-toggle">☰</button>
              <div className="model-selector">Biopsy AI <span style={{ color: "var(--claude-accent)" }}>v2.1</span></div>
           </header>

           <div className="chat-scroll-area" ref={scrollRef}>
             {messages.length === 0 ? (
               <div className="hero-greeting">
                 <div className="hero-star">✴️</div>
                 <h1>Welcome to Biopsy AI assistant.</h1>
                 <p style={{ color: "var(--claude-text-muted)", marginBottom: "30px" }}>Upload a biopsy sample to perform classification and lesion segmentation.</p>
                 <div className="hero-suggestions">
                    <button onClick={() => fileInputRef.current?.click()}>Upload Biopsy (New)</button>
                    <button>Documentation</button>
                 </div>
               </div>
             ) : (
               <div className="message-list">
                 {messages.map((msg) => (
                   <div key={msg.id} className={`message-row ${msg.role}`}>
                      <div className="message-avatar">
                        {msg.role === "assistant" ? "✴️" : "IA"}
                      </div>
                      <div className="message-content">
                        <div className="message-text">{msg.content}</div>
                        {msg.image && (
                           <div className="message-image-card">
                              <img src={msg.image} alt="User Upload" />
                              <div className="image-label">{msg.fileName}</div>
                           </div>
                        )}
                        {msg.results && (
                           <button className="artifact-pill" onClick={() => setSelectedArtifact(msg.results!)}>
                              📊 Result Artifact
                           </button>
                        )}
                        {msg.status === "loading" && <div className="typing-indicator"><span>.</span><span>.</span><span>.</span></div>}
                      </div>
                   </div>
                 ))}
               </div>
             )}
           </div>

           <footer className="chat-footer">
             <div className="input-container">
               <button className="attach-btn" onClick={() => fileInputRef.current?.click()}>+</button>
               <input 
                 type="text" 
                 placeholder="Upload biopsy or ask questions..." 
                 value={inputValue}
                 onChange={(e) => setInputValue(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && setInputValue("")}
               />
               <button className="send-btn" disabled={!inputValue.trim()}>↑</button>
               <input 
                 type="file" 
                 ref={fileInputRef} 
                 style={{ display: "none" }} 
                 onChange={handleFileUpload} 
                 accept="image/*"
               />
             </div>
             <div className="footer-disclaimer">Biopsy AI. Minimalist Diagnostic Tool.</div>
           </footer>
        </section>

        {/* Artifact Panel Section */}
        {selectedArtifact && (
          <aside className="artifact-panel">
            <div className="artifact-header">
               <div className="header-left">
                  <span className="close-btn" onClick={() => setSelectedArtifact(null)}>✕</span>
                  <div className="artifact-meta">
                     <div className="artifact-title">Biopsy_Analysis_Artifact</div>
                     <div className="artifact-subtitle">Classification and Segmentation Preview</div>
                  </div>
               </div>
               <div className="header-right">
                  <div className="badge-pill">Code</div>
                  <div className="badge-pill active">Preview</div>
               </div>
            </div>

            <div className="artifact-body">
               <div className="result-hero" style={{ background: "var(--claude-bg-dark)", border: "1px solid var(--claude-border)" }}>
                  <div className="result-info">
                     <div className="info-label" style={{ color: "var(--claude-text-muted)", fontSize: "0.75rem", marginBottom: "4px" }}>Classification Diagnosis</div>
                     <div className="result-badge" style={{ fontSize: "1.2rem", fontWeight: "600", color: "var(--claude-accent)" }}>
                        {selectedArtifact.pred_class_name_ru || `Class ${selectedArtifact.pred_class}`}
                     </div>
                  </div>
                  <div className="result-metrics" style={{ textAlign: "right" }}>
                     <div className="info-label" style={{ color: "var(--claude-text-muted)", fontSize: "0.75rem", marginBottom: "4px" }}>Model Confidence</div>
                     <div className="result-confidence" style={{ fontWeight: "600" }}>
                        {(selectedArtifact.confidence || 0) * 100}%
                     </div>
                  </div>
               </div>

               <div className="preview-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "10px", marginTop: "20px" }}>
                  {selectedArtifact.segmentation?.maps.overlay && (
                    <div className="preview-card">
                       <label>Binary Overlay</label>
                       <div className="img-container">
                          <img src={`data:image/png;base64,${selectedArtifact.segmentation.maps.overlay.png_b64}`} alt="Overlay" />
                       </div>
                    </div>
                  )}
                  {selectedArtifact.segmentation?.maps.mask && (
                    <div className="preview-card">
                       <label>Binary Mask (Raw)</label>
                       <div className="img-container">
                          <img src={`data:image/png;base64,${selectedArtifact.segmentation.maps.mask.png_b64}`} alt="Mask" />
                       </div>
                    </div>
                  )}
                  {selectedArtifact.segmentation?.maps.contour && (
                    <div className="preview-card">
                       <label>Contour Extraction</label>
                       <div className="img-container">
                          <img src={`data:image/png;base64,${selectedArtifact.segmentation.maps.contour.png_b64}`} alt="Contour" />
                       </div>
                    </div>
                  )}
                  {selectedArtifact.segmentation?.maps.prob && (
                    <div className="preview-card">
                       <label>Probability Map (Heat)</label>
                       <div className="img-container">
                          <img src={`data:image/png;base64,${selectedArtifact.segmentation.maps.prob.png_b64}`} alt="Prob" />
                       </div>
                    </div>
                  )}
                  {selectedArtifact.segmentation?.maps.distance && (
                    <div className="preview-card">
                       <label>Distance Transform</label>
                       <div className="img-container">
                          <img src={`data:image/png;base64,${selectedArtifact.segmentation.maps.distance.png_b64}`} alt="Distance" />
                       </div>
                    </div>
                  )}
               </div>

               {selectedArtifact.segmentation && (
                 <div className="metrics-card" style={{ marginTop: "20px", padding: "15px", borderRadius: "8px", background: "rgba(0,0,0,0.2)", border: "1px solid var(--claude-border)" }}>
                    <label style={{ fontSize: "0.85rem", color: "var(--claude-text-muted)", display: "block", marginBottom: "10px" }}>Segmentation Statistics</label>
                    <div className="metrics-list" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px" }}>
                       <div className="metric-item">
                          <span>Lesion Coverage:</span>
                          <strong>{(selectedArtifact.segmentation.coverage * 100).toFixed(2)}%</strong>
                       </div>
                       <div className="metric-item">
                          <span>Active Components:</span>
                          <strong>{selectedArtifact.segmentation.num_components}</strong>
                       </div>
                       <div className="metric-item">
                          <span>Used Threshold:</span>
                          <strong>{selectedArtifact.segmentation.thr_used.toFixed(4)}</strong>
                       </div>
                       <div className="metric-item">
                          <span>Domain Reason:</span>
                          <strong style={{ fontSize: "0.75rem" }}>{selectedArtifact.reason}</strong>
                       </div>
                    </div>
                 </div>
               )}
            </div>
          </aside>
        )}

      </main>
    </div>
  );
}



