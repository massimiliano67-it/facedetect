import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000/api';

function App() {
  // --- ESTADOS DE LA APLICACIÓN ---
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'manage'
  const [files, setFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Datos de clusters: { "0": {name: "Juan", faces: [...]}, ... }
  const [clusters, setClusters] = useState({});
  
  // Estado para la gestión (Pestaña Manage)
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [nameInput, setNameInput] = useState("");
  
  // Estado para selección múltiple (Batch Delete)
  const [selectedForBatch, setSelectedForBatch] = useState(new Set());

  // Estado del Video en segundo plano
  const [videoStatus, setVideoStatus] = useState(null);

  // --- EFECTOS (POLLING) ---
  
  // Verificar estado del video cada 2 segundos
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/video-status`);
        setVideoStatus(res.data);

        // Si el video acaba de terminar (estaba procesando y ahora no), recargamos datos
        if (res.data.faces_found > 0 && !res.data.processing && videoStatus?.processing) {
          alert(`✅ Video procesado. Se encontraron ${res.data.faces_found} caras nuevas.`);
          loadClusters();
        }
      } catch (e) {
        // Ignorar errores de conexión silenciosamente en polling
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [videoStatus?.processing]); // Dependencia para comparar estado anterior

  // --- FUNCIONES DE CARGA ---

  const loadClusters = async () => {
    try {
      const res = await axios.get(`${API_URL}/clusters`);
      setClusters(res.data);
      return res.data;
    } catch (error) {
      console.error("Error cargando clusters:", error);
      return {};
    }
  };

  const handleUpload = async () => {
    if (!files) return alert("⚠️ Selecciona archivos primero");
    setLoading(true);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();

        if (file.type.startsWith('video')) {
           // --- SUBIDA DE VIDEO ---
           formData.append('file', file);
           await axios.post(`${API_URL}/cluster-video`, formData);
           alert("🎥 El video se está procesando en segundo plano. Puedes continuar usando la app.");
        } else {
           // --- SUBIDA DE IMAGEN ---
           // Nota: El backend espera una lista 'files', enviamos uno por uno para simplificar loop
           formData.append('files', file);
           await axios.post(`${API_URL}/cluster-faces`, formData);
        }
      }
      
      // Recargar datos actualizados
      const newData = await loadClusters();
      
      // Cambiar a pestaña de gestión
      setActiveTab('manage');
      setFiles(null);

      // Auto-seleccionar el primer cluster si no hay uno seleccionado
      const keys = Object.keys(newData);
      if (keys.length > 0 && !selectedClusterId) {
        setSelectedClusterId(keys[0]);
        setNameInput(newData[keys[0]].name);
      }

    } catch (error) {
      console.error(error);
      alert("❌ Error al subir archivos. Revisa la consola.");
    } finally {
      setLoading(false);
    }
  };

  // --- FUNCIONES DE GESTIÓN ---

  const handleRename = async () => {
    if (!selectedClusterId) return;
    try {
      const res = await axios.post(`${API_URL}/rename-cluster`, {
        cluster_id: selectedClusterId,
        new_name: nameInput
      });
      setClusters(res.data);
    } catch (error) {
      alert("Error al renombrar");
    }
  };

  const handleMoveFace = async (faceId, targetClusterId) => {
    try {
      let finalTargetId = targetClusterId;
      let newName = null;

      // Lógica para crear nuevo grupo
      if (targetClusterId === 'new_group') {
        finalTargetId = 'manual_group_' + Date.now();
        newName = "Nuevo Grupo";
      }

      const res = await axios.post(`${API_URL}/move-face`, {
        face_id: faceId,
        to_cluster_id: finalTargetId,
        new_cluster_name: newName
      });
      
      setClusters(res.data);
      refreshSelection(res.data);

    } catch (error) {
      alert("Error al mover imagen");
    }
  };

  const handleDeleteFace = async (faceId) => {
    if (!window.confirm("¿Estás seguro de eliminar esta foto?")) return;
    try {
      const res = await axios.post(`${API_URL}/delete-face`, { face_id: faceId });
      setClusters(res.data);
      refreshSelection(res.data);
    } catch (error) {
      alert("Error al eliminar foto");
    }
  };

  const handleDeleteCluster = async () => {
    if (!selectedClusterId) return;
    const confirmMsg = `⚠️ ¿Borrar a "${clusters[selectedClusterId].name}" y TODAS sus fotos?\nEsta acción es irreversible.`;
    if (!window.confirm(confirmMsg)) return;

    try {
      const res = await axios.post(`${API_URL}/delete-cluster`, { cluster_id: selectedClusterId });
      setClusters(res.data);
      
      // Forzar selección a null o al primero
      const keys = Object.keys(res.data);
      if (keys.length > 0) {
          setSelectedClusterId(keys[0]);
          setNameInput(res.data[keys[0]].name);
      } else {
          setSelectedClusterId(null);
          setNameInput("");
      }
    } catch (error) {
      alert("Error al eliminar grupo");
    }
  };

  // --- BATCH DELETE LOGIC ---
  
  const toggleBatchSelection = (e, clusterId) => {
    e.stopPropagation(); // Evitar que seleccione el cluster como activo
    const newSet = new Set(selectedForBatch);
    if (newSet.has(clusterId)) {
      newSet.delete(clusterId);
    } else {
      newSet.add(clusterId);
    }
    setSelectedForBatch(newSet);
  };

  const handleBatchDelete = async () => {
    if (selectedForBatch.size === 0) return;
    
    const confirmMsg = `⚠️ ¿Estás seguro de eliminar ${selectedForBatch.size} grupos seleccionados?\nSe borrarán TODAS sus fotos.`;
    if (!window.confirm(confirmMsg)) return;

    try {
      const res = await axios.post(`${API_URL}/delete-clusters`, { 
        cluster_ids: Array.from(selectedForBatch) 
      });
      
      setClusters(res.data);
      setSelectedForBatch(new Set()); // Limpiar selección
      refreshSelection(res.data);

    } catch (error) {
      console.error(error);
      alert("Error al eliminar grupos seleccionados");
    }
  };

  // Helper para mantener la UI consistente si el grupo actual desaparece
  const refreshSelection = (newData) => {
    if (!newData[selectedClusterId]) {
      const keys = Object.keys(newData);
      if (keys.length > 0) {
          setSelectedClusterId(keys[0]);
          setNameInput(newData[keys[0]].name);
      } else {
          setSelectedClusterId(null);
          setNameInput("");
      }
    }
  };

  // Helper para color del badge
  const getConfidenceColor = (score) => {
    if (score >= 80) return '#2ecc71'; // Verde
    if (score >= 50) return '#f1c40f'; // Amarillo
    return '#e74c3c';                 // Rojo
  };

  // --- RENDERIZADO ---
  
  const clusterList = Object.values(clusters);
  const selectedCluster = clusters[selectedClusterId];

  return (
    <div className="app-container">
      
      {/* HEADER */}
      <header className="app-header">
        <h1>🕵️ Reconocimiento Facial IA</h1>
        <div className="tabs">
          <button 
            className={activeTab === 'upload' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('upload')}
          >
            📥 Subir
          </button>
          <button 
            className={activeTab === 'manage' ? 'tab active' : 'tab'}
            onClick={() => {
                setActiveTab('manage');
                if(!selectedClusterId && clusterList.length > 0) {
                    loadClusters().then(data => {
                        const keys = Object.keys(data);
                        if(keys.length > 0) {
                            setSelectedClusterId(keys[0]);
                            setNameInput(data[keys[0]].name);
                        }
                    });
                }
            }}
          >
            👥 Gestionar
          </button>
        </div>
      </header>

      {/* BARRA DE ESTADO DE VIDEO */}
      {videoStatus && videoStatus.processing && (
        <div className="video-progress-bar">
           <div className="spinner"></div>
           <span>
             🎥 Procesando <b>{videoStatus.filename}</b> | 
             Frame: {videoStatus.frames_processed} | 
             Caras encontradas: {videoStatus.faces_found}
           </span>
        </div>
      )}

      {/* VISTA UPLOAD */}
      {activeTab === 'upload' && (
        <div className="tab-content upload-view">
          <div className="upload-zone">
            <input 
              type="file" 
              multiple 
              accept="image/*, video/*" 
              onChange={(e) => setFiles(e.target.files)} 
            />
            <p>
                {files 
                  ? `📂 ${files.length} archivo(s) seleccionado(s)` 
                  : "Arrastra fotos o videos aquí"}
            </p>
          </div>
          <button 
            className="btn-primary" 
            onClick={handleUpload} 
            disabled={loading || !files}
          >
            {loading ? "⏳ Procesando..." : "🚀 Analizar Archivos"}
          </button>
        </div>
      )}

      {/* VISTA GESTIÓN */}
      {activeTab === 'manage' && (
        <div className="tab-content manage-view">
          
          {/* SIDEBAR LISTA */}
          <aside className="sidebar">
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              padding: '20px', 
              borderBottom: '1px solid var(--border-color)' 
            }}>
              <h3 style={{ padding: 0, border: 'none', margin: 0 }}>Personas ({clusterList.length})</h3>
              {selectedForBatch.size > 0 && (
                <button 
                  onClick={handleBatchDelete}
                  style={{
                    background: 'var(--danger)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    fontWeight: 'bold'
                  }}
                >
                  Borrar ({selectedForBatch.size})
                </button>
              )}
            </div>
            
            <ul>
              {clusterList.map((c) => (
                <li 
                  key={c.id}
                  className={selectedClusterId === c.id ? 'active' : ''}
                  onClick={() => { 
                    setSelectedClusterId(c.id); 
                    setNameInput(c.name); 
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <input 
                      type="checkbox" 
                      checked={selectedForBatch.has(c.id)}
                      onClick={(e) => toggleBatchSelection(e, c.id)}
                      style={{ cursor: 'pointer' }}
                    />
                    <span>{c.name}</span>
                  </div>
                  <span className="badge">{c.faces.length}</span>
                </li>
              ))}
            </ul>
          </aside>

          {/* AREA PRINCIPAL */}
          <main className="main-area">
            {selectedCluster ? (
              <>
                <div className="cluster-actions-header">
                  <div className="input-group">
                      <input 
                        type="text" 
                        value={nameInput} 
                        onChange={(e) => setNameInput(e.target.value)}
                        placeholder="Nombre de la persona"
                      />
                      <button className="btn-save" onClick={handleRename}>Guardar Nombre</button>
                  </div>
                  
                  <button className="btn-delete-group" onClick={handleDeleteCluster}>
                    🗑️ Eliminar Grupo
                  </button>
                </div>

                <div className="faces-grid">
                  {selectedCluster.faces.map((face) => (
                    <div key={face.id} className="face-card">
                      <div className="image-container">
                        <img 
                          src={`data:image/jpeg;base64,${face.image}`} 
                          alt="face" 
                        />
                        
                        {/* Botón Eliminar Foto */}
                        <button 
                            className="btn-delete-face" 
                            onClick={() => handleDeleteFace(face.id)}
                            title="Eliminar foto"
                        >
                            ×
                        </button>

                        {/* Badge de Confianza */}
                        <div 
                          className="confidence-badge" 
                          style={{ backgroundColor: getConfidenceColor(face.confidence) }}
                        >
                          {face.confidence}%
                        </div>
                      </div>
                      
                      <div className="card-controls">
                        <select 
                          value={selectedClusterId} 
                          onChange={(e) => handleMoveFace(face.id, e.target.value)}
                        >
                          <option disabled value={selectedClusterId}>Mover a...</option>
                          <option value="new_group">➕ Nuevo Grupo</option>
                          {clusterList.map(c => (
                             c.id !== selectedClusterId && (
                               <option key={c.id} value={c.id}>{c.name}</option>
                             )
                          ))}
                        </select>
                      </div>
                      <div className="filename" title={face.filename}>{face.filename}</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>👈 Selecciona una persona del menú</p>
                <small>O sube nuevas fotos en la pestaña "Subir"</small>
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}

export default App;