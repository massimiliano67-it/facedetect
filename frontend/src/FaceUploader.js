import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // --- ESTADOS ---
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'manage'
  const [files, setFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Datos principales: { "clusterID": { name: "Juan", faces: [...] } }
  const [clusters, setClusters] = useState({}); 
  
  // Estados de la vista de gestión
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [nameInput, setNameInput] = useState("");

  // --- FUNCIONES ---

  const handleUpload = async () => {
    if (!files) return alert("Selecciona archivos primero");
    setLoading(true);
    
    try {
      // Vamos a separar imágenes de videos o procesar uno por uno
      // Para simplificar, asumamos que si hay un video, lo procesamos aparte
      // o enviamos todo al endpoint correspondiente.
      
      // ESTRATEGIA: Iterar archivos y enviar según tipo
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append(file.type.startsWith('video') ? 'file' : 'files', file);

        if (file.type.startsWith('video')) {
           // Es un VIDEO -> Endpoint de Video
           // Nota: Video suele ser pesado, mejor enviar uno por uno y esperar
           await axios.post('http://localhost:8000/api/cluster-video', formData);
        } else {
           // Es una IMAGEN -> Endpoint de Imágenes
           // Nota: El endpoint de imágenes espera una lista 'files', aquí mandamos 1 a 1 
           // o podrías agruparlas. Para no romper tu código anterior, 
           // si es imagen usaremos la logica de lote si quieres, 
           // pero aquí un ejemplo simple enviando archivo:
           
           // Corrección para mantener compatibilidad con tu endpoint anterior
           // que espera una lista:
           const imgForm = new FormData();
           imgForm.append('files', file);
           await axios.post('http://localhost:8000/api/cluster-faces', imgForm);
        }
      }

      // Al terminar de subir todo, pedimos el estado final (truco para refrescar)
      // Podrías hacer un endpoint GET /api/clusters o simplemente
      // llamar a cualquiera de los anteriores (el ultimo retorno tendrá la data)
      
      // Para asegurar que tenemos la data fresca, haremos una llamada 'dummy' 
      // o simplemente confiamos en que la última respuesta del loop trajo la data.
      // Una forma limpia es tener un endpoint GET. 
      // PERO, vamos a usar el endpoint de Rename (con datos dummy) 
      // o Move para refrescar, o crear un GET simple.
      
      // Creemos un GET simple en backend o usemos el ultimo response
      // Hack rapido: volvemos a llamar rename con id invalido solo para recibir data actualizada
      // O MEJOR: Crear endpoint GET en main.py (ver abajo)

      const res = await axios.get('http://localhost:8000/api/clusters');
      setClusters(res.data);
      
      setActiveTab('manage');
      setFiles(null);
      
      // Auto seleccionar
      const keys = Object.keys(res.data);
      if (keys.length > 0) {
        setSelectedClusterId(keys[0]);
        setNameInput(res.data[keys[0]].name);
      }

    } catch (error) {
      console.error(error);
      alert("Error al procesar (Nota: Los videos largos pueden tardar)");
    } finally {
      setLoading(false);
    }
  };

  const handleRename = async () => {
    if (!selectedClusterId) return;
    try {
      const res = await axios.post('http://localhost:8000/api/rename-cluster', {
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
      // Si es "new_group", generamos un ID aleatorio en el cliente o enviamos flag
      let finalTargetId = targetClusterId;
      let newName = null;

      if (targetClusterId === 'new_group') {
        finalTargetId = 'manual_group_' + Date.now();
        newName = "Nuevo Grupo";
      }

      const res = await axios.post('http://localhost:8000/api/move-face', {
        face_id: faceId,
        to_cluster_id: finalTargetId,
        new_cluster_name: newName
      });
      
      setClusters(res.data);

      // Si el grupo actual se vació y desapareció, seleccionar otro
      if (!res.data[selectedClusterId]) {
         const keys = Object.keys(res.data);
         if(keys.length > 0) setSelectedClusterId(keys[0]);
         else setSelectedClusterId(null);
      }

    } catch (error) {
      console.error(error);
      alert("Error al mover imagen");
    }
  };

  // Función para borrar una foto individual
  const handleDeleteFace = async (faceId) => {
    if (!window.confirm("¿Estás seguro de eliminar esta foto?")) return;
    
    try {
      const res = await axios.post('http://localhost:8000/api/delete-face', {
        face_id: faceId
      });
      setClusters(res.data);
      
      // Si el cluster actual se queda vacío tras borrar, seleccionamos otro o null
      if (!res.data[selectedClusterId]) {
        const keys = Object.keys(res.data);
        setSelectedClusterId(keys.length > 0 ? keys[0] : null);
      }
    } catch (error) {
      alert("Error al eliminar foto");
    }
  };

  // Función para borrar todo el cluster
  const handleDeleteCluster = async () => {
    if (!selectedClusterId) return;
    const confirmMsg = `¿⚠️ CUIDADO: Estás a punto de borrar el grupo "${clusters[selectedClusterId].name}" y TODAS sus fotos?\n\nEsta acción no se puede deshacer.`;
    if (!window.confirm(confirmMsg)) return;

    try {
      const res = await axios.post('http://localhost:8000/api/delete-cluster', {
        cluster_id: selectedClusterId
      });
      setClusters(res.data);
      
      // Como borramos el cluster actual, forzamos la selección al primero disponible (o null)
      const keys = Object.keys(res.data);
      setSelectedClusterId(keys.length > 0 ? keys[0] : null);
      
      // Actualizar input de nombre si hay nuevo cluster, sino limpiar
      if (keys.length > 0) setNameInput(res.data[keys[0]].name);
      else setNameInput("");

    } catch (error) {
      alert("Error al eliminar cluster");
    }
  };

  // Helpers para renderizado
  const clusterList = Object.values(clusters);
  const selectedCluster = clusters[selectedClusterId];
// Función para determinar color según porcentaje
const getConfidenceColor = (score) => {
  if (score >= 80) return '#2ecc71'; // Verde (Alta confianza)
  if (score >= 50) return '#f1c40f'; // Amarillo (Media)
  return '#e74c3c';                 // Rojo (Baja/Dudosa)
};

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="app-header">
        <h1>🤖 Face Cluster AI</h1>
        <div className="tabs">
          <button 
            className={activeTab === 'upload' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('upload')}
          >
            📥 Subir
          </button>
          <button 
            className={activeTab === 'manage' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('manage')}
            disabled={clusterList.length === 0}
          >
            👥 Gestionar
          </button>
        </div>
      </header>

      {/* TAB 1: UPLOAD */}
      {activeTab === 'upload' && (
        <div className="tab-content upload-view">
          <div className="upload-zone">
<input 
  type="file" 
  multiple 
  accept="image/*, video/*"  // <--- ACEPTAR AMBOS
  onChange={(e) => setFiles(e.target.files)} 
/>
            <p>{files ? `${files.length} fotos listas` : "Selecciona tus fotos aquí"}</p>
          </div>
          <button 
            className="btn-primary" 
            onClick={handleUpload} 
            disabled={loading || !files}
          >
            {loading ? "Procesando..." : "Analizar Imágenes"}
          </button>
        </div>
      )}

      {/* TAB 2: MANAGE */}
      {/* TAB 2: MANAGE */}
      {activeTab === 'manage' && (
        <div className="tab-content manage-view">
          
          <aside className="sidebar">
            {/* ... (código del sidebar igual que antes) ... */}
            <h3>Personas ({clusterList.length})</h3>
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
                    <span>{c.name}</span>
                    <span className="badge">{c.faces.length}</span>
                </li>
                ))}
            </ul>
          </aside>

          <main className="main-area">
            {selectedCluster ? (
              <>
                <div className="cluster-actions">
                  <input 
                    type="text" 
                    value={nameInput} 
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder="Nombre de la persona"
                  />
                  <button className="btn-save" onClick={handleRename}>Guardar</button>
                  
                  {/* --- NUEVO BOTÓN BORRAR CLUSTER --- */}
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
                        
                        {/* --- NUEVO BOTÓN BORRAR FOTO (X) --- */}
                        <button 
                            className="btn-delete-face" 
                            onClick={() => handleDeleteFace(face.id)}
                            title="Eliminar esta foto"
                        >
                            ×
                        </button>

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
              <div className="empty-state">Selecciona una persona del menú o sube fotos</div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}

export default App;