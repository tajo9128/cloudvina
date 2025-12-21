import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import PreparationProgress from '../components/PreparationProgress'
import { Upload, FileText, Database, Cpu, Play, CheckCircle2, AlertCircle, ArrowRight, FlaskConical } from 'lucide-react'
import { trackEvent } from '../services/analytics' // Import Analytics

export default function BatchDockingPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFiles, setLigandFiles] = useState([])
    const [csvFile, setCsvFile] = useState(null)
    const [error, setError] = useState(null)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [batchId, setBatchId] = useState(null)
    const [preparationStep, setPreparationStep] = useState(0)

    // Upload Mode Toggle
    const [uploadMode, setUploadMode] = useState('files') // 'files' or 'csv'

    // Engine State (Hardcoded to Consensus now, but keeping var for API compat if needed)
    // Actually, we can just pass 'consensus' directly.

    // --- SUBMISSION HANDLERS ---
    const handleFilesSubmit = async () => {
        if (!receptorFile || ligandFiles.length === 0) {
            setError('Please upload a receptor and at least one ligand.')
            return
        }

        setLoading(true)
        setError(null)

        try {
            // 1. Create Batch (Files Mode)
            const { data: { session } } = await supabase.auth.getSession()
            const initRes = await fetch(`${API_URL}/jobs/batch/submit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filenames: ligandFiles.map(f => f.name)
                })
            })

            if (!initRes.ok) throw new Error('Failed to initialize batch')
            const initData = await initRes.json()
            const newBatchId = initData.batch_id
            setBatchId(newBatchId)

            // 2. Upload Files
            setUploadProgress(10)

            // Upload Receptor
            await fetch(initData.upload_urls.receptor_url, {
                method: 'PUT',
                body: receptorFile
            })
            setUploadProgress(30)

            // Upload Ligands (Parallel)
            let completed = 0
            const total = ligandFiles.length

            // Batch requests for stability
            const CHUNK_SIZE = 5
            const ligandUrlMap = initData.upload_urls.ligands.reduce((acc, curr) => ({ ...acc, [curr.filename]: curr.url }), {})

            for (let i = 0; i < total; i += CHUNK_SIZE) {
                const chunk = ligandFiles.slice(i, i + CHUNK_SIZE)
                await Promise.all(chunk.map(async file => {
                    const url = ligandUrlMap[file.name]
                    if (url) {
                        await fetch(url, { method: 'PUT', body: file })
                    }
                    completed++
                    setUploadProgress(30 + Math.round((completed / total) * 60))
                }))
            }

            // 3. Start Batch (Engine: Consensus Default)
            const startRes = await fetch(`${API_URL}/jobs/batch/${newBatchId}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    grid_params: {
                        center_x: 0, center_y: 0, center_z: 0,
                        size_x: 20, size_y: 20, size_z: 20
                    },
                    engine: 'consensus' // FORCE CONSENSUS
                })
            })

            if (!startRes.ok) throw new Error('Failed to start batch processing')

            trackEvent('batch_docking:started', { batch_id: newBatchId, engine: 'consensus' })

            // Redirect
            navigate(`/dock/batch/${newBatchId}`)

        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const handleCsvSubmit = async () => {
        if (!receptorFile || !csvFile) {
            setError('Please upload a receptor and a CSV file.')
            return
        }

        setLoading(true)
        setError(null)
        setUploadProgress(10)

        try {
            const formData = new FormData()
            formData.append('receptor_file', receptorFile)
            formData.append('csv_file', csvFile)
            // Use defaults for grid
            formData.append('grid_center_x', 0)
            formData.append('grid_center_y', 0)
            formData.append('grid_center_z', 0)
            formData.append('grid_size_x', 20)
            formData.append('grid_size_y', 20)
            formData.append('grid_size_z', 20)
            formData.append('engine', 'consensus') // FORCE CONSENSUS

            const { data: { session } } = await supabase.auth.getSession()
            const res = await fetch(`${API_URL}/jobs/batch/submit-csv`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: formData
            })

            setUploadProgress(100)

            if (!res.ok) {
                const errorData = await res.json()
                throw new Error(errorData.detail || 'CSV Batch Submission Failed')
            }

            const data = await res.json()
            trackEvent('batch_docking:csv_started', { batch_id: data.batch_id, engine: 'consensus' })
            navigate(`/dock/batch/${data.batch_id}`)

        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 pb-24">
            {/* Header */}
            <div className="bg-white border-b border-slate-200 sticky top-0 z-30">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/dashboard" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowRight className="w-5 h-5 rotate-180" />
                        </Link>
                        <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                            <FlaskConical className="w-5 h-5 text-indigo-600" />
                            New Docking Experiment
                        </h1>
                    </div>
                    <div className="flex bg-slate-100 p-1 rounded-lg">
                        <button
                            onClick={() => setUploadMode('files')}
                            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${uploadMode === 'files' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                        >
                            Files
                        </button>
                        <button
                            onClick={() => setUploadMode('csv')}
                            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${uploadMode === 'csv' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                        >
                            CSV (SMILES)
                        </button>
                    </div>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

                {/* 1. Receptor Upload */}
                <div className="mb-8 animate-fade-in-up" style={{ animationDelay: '0ms' }}>
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">1</div>
                        <h2 className="text-lg font-bold text-slate-900">Step 1: Target Receptor</h2>
                    </div>

                    <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:border-indigo-300 transition-colors relative group">
                        {!receptorFile ? (
                            <div className="flex flex-col items-center justify-center py-8">
                                <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                    <Database className="w-8 h-8 text-indigo-500" />
                                </div>
                                <h3 className="text-lg font-medium text-slate-900 mb-2">Upload PDB Structure</h3>
                                <p className="text-slate-500 text-center max-w-sm mb-6">
                                    Drag and drop your prepared receptor file here. Supported formats: .pdb, .pdbqt
                                </p>
                                <label className="btn-primary cursor-pointer">
                                    Select Receptor
                                    <input
                                        type="file"
                                        className="hidden"
                                        accept=".pdb,.pdbqt"
                                        onChange={(e) => setReceptorFile(e.target.files[0])}
                                    />
                                </label>
                            </div>
                        ) : (
                            <div className="flex items-center justify-between bg-indigo-50/50 p-4 rounded-xl border border-indigo-100">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 bg-white rounded-lg shadow-sm">
                                        <Database className="w-6 h-6 text-indigo-600" />
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-900">{receptorFile.name}</div>
                                        <div className="text-xs text-slate-500">{(receptorFile.size / 1024).toFixed(1)} KB</div>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setReceptorFile(null)}
                                    className="p-2 hover:bg-red-50 text-slate-400 hover:text-red-500 rounded-lg transition-colors"
                                >
                                    ✕
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* 2. Ligand Upload / CSV */}
                <div className="mb-8 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">2</div>
                        <h2 className="text-lg font-bold text-slate-900">
                            {uploadMode === 'files' ? 'Step 2: Ligand Library' : 'Step 2: SMILES Data'}
                        </h2>
                    </div>

                    <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm hover:border-indigo-300 transition-colors">
                        {uploadMode === 'files' ? (
                            <>
                                {ligandFiles.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center py-8">
                                        <div className="w-16 h-16 bg-purple-50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                            <FileText className="w-8 h-8 text-purple-500" />
                                        </div>
                                        <p className="text-slate-500 text-center max-w-sm mb-6">
                                            Upload multiple ligand files for batch screening. Supported: .pdbqt, .sdf, .mol2
                                        </p>
                                        <label className="btn-secondary cursor-pointer">
                                            Select Files
                                            <input
                                                type="file"
                                                multiple
                                                className="hidden"
                                                accept=".pdbqt,.sdf,.mol2"
                                                onChange={(e) => setLigandFiles(Array.from(e.target.files))}
                                            />
                                        </label>
                                    </div>
                                ) : (
                                    <div>
                                        <div className="flex justify-between items-center mb-4">
                                            <h3 className="font-bold text-slate-700">{ligandFiles.length} Ligands Selected</h3>
                                            <button onClick={() => setLigandFiles([])} className="text-xs text-red-500 hover:underline">Clear All</button>
                                        </div>
                                        <div className="max-h-48 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                                            {ligandFiles.map((file, i) => (
                                                <div key={i} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100 text-sm">
                                                    <span className="truncate max-w-[200px] text-slate-700">{file.name}</span>
                                                    <span className="text-xs text-slate-400">{(file.size / 1024).toFixed(1)} KB</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="flex flex-col items-center justify-center py-8">
                                {!csvFile ? (
                                    <>
                                        <div className="w-16 h-16 bg-green-50 rounded-full flex items-center justify-center mb-4">
                                            <FileText className="w-8 h-8 text-green-500" />
                                        </div>
                                        <p className="text-slate-500 text-center max-w-sm mb-6">
                                            Upload a CSV file containing a 'smiles' column.
                                        </p>
                                        <label className="btn-secondary cursor-pointer">
                                            Select CSV
                                            <input
                                                type="file"
                                                className="hidden"
                                                accept=".csv"
                                                onChange={(e) => setCsvFile(e.target.files[0])}
                                            />
                                        </label>
                                    </>
                                ) : (
                                    <div className="w-full flex items-center justify-between bg-green-50/50 p-4 rounded-xl border border-green-100">
                                        <span className="font-bold text-green-800">{csvFile.name}</span>
                                        <button onClick={() => setCsvFile(null)} className="text-green-600 hover:text-green-800">✕</button>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* 3. Engine Selection (Hardcoded Visual) */}
                <div className="mb-8 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">3</div>
                        <h2 className="text-lg font-bold text-slate-900">Step 3: Docking Engine</h2>
                    </div>

                    <div className="grid grid-cols-1 gap-6">
                        <div className="relative p-6 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl border-2 border-indigo-500 shadow-lg flex items-start gap-4 cursor-default">
                            <div className="absolute top-4 right-4 animate-pulse">
                                <div className="bg-indigo-600/10 text-indigo-600 text-[10px] font-bold px-2 py-1 rounded-full uppercase tracking-wider border border-indigo-200">
                                    Recommended
                                </div>
                            </div>
                            <div className="p-4 bg-white rounded-xl shadow-sm self-center">
                                <Cpu className="w-8 h-8 text-indigo-600" />
                            </div>
                            <div className="flex-1">
                                <h3 className="text-lg font-bold text-slate-900 mb-1">Consensus Mode</h3>
                                <p className="text-sm text-slate-600 leading-relaxed mb-3">
                                    Combines <strong>AutoDock Vina</strong> (classic scoring) with <strong>Gnina</strong> (Deep Learning CNN) for maximum accuracy.
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    <span className="px-2 py-1 bg-white rounded border border-indigo-100 text-xs font-mono text-indigo-600">Vina 1.2.3</span>
                                    <span className="px-2 py-1 bg-white rounded border border-purple-100 text-xs font-mono text-purple-600">Gnina CNN</span>
                                </div>
                            </div>
                            <div className="self-center">
                                <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center">
                                    <CheckCircle2 className="w-4 h-4 text-white" />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>


                {/* Submit Action */}
                <div className="flex justify-end pt-8 border-t border-slate-200">
                    {error && (
                        <div className="mr-auto flex items-center gap-2 text-red-600 bg-red-50 px-4 py-2 rounded-lg">
                            <AlertCircle className="w-5 h-5" />
                            <span className="text-sm font-medium">{error}</span>
                        </div>
                    )}

                    <button
                        onClick={uploadMode === 'files' ? handleFilesSubmit : handleCsvSubmit}
                        disabled={loading || !receptorFile}
                        className={`
                             relative overflow-hidden group px-8 py-4 bg-slate-900 text-white font-bold rounded-xl shadow-xl 
                             transition-all transform hover:-translate-y-1 hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
                         `}
                    >
                        <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-indigo-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <div className="relative flex items-center gap-3">
                            {loading ? (
                                <>
                                    <span className="animate-spin text-xl">⚪</span>
                                    <span>Initializing Job... {uploadProgress > 0 && `${uploadProgress}%`}</span>
                                </>
                            ) : (
                                <>
                                    <span>Launch Experiment</span>
                                    <Play className="w-5 h-5 fill-current" />
                                </>
                            )}
                        </div>
                    </button>
                </div>
            </div>

            {/* Loading Overlay */}
            {loading && (
                <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
                    <PreparationProgress step={preparationStep} progress={uploadProgress} />
                </div>
            )}
        </div>
    )
}
