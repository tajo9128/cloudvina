import { useState, useEffect, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import PreparationProgress from '../components/PreparationProgress'
import { Upload, FileText, Database, Cpu, Play, CheckCircle2, AlertCircle, ArrowRight, FlaskConical, Terminal, Activity, Server, ShieldCheck, FileCheck, Check, Info } from 'lucide-react'
import { trackEvent } from '../services/analytics'

export default function BatchDockingPage() {
    const navigate = useNavigate()

    // Core State
    const [loading, setLoading] = useState(false)
    const [processingStage, setProcessingStage] = useState('idle') // 'idle' | 'uploading' | 'processing' | 'complete'
    const [logs, setLogs] = useState([])
    const [uploadProgress, setUploadProgress] = useState(0)

    // Form State
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFiles, setLigandFiles] = useState([])
    const [csvFile, setCsvFile] = useState(null)
    const [error, setError] = useState(null)
    const [batchId, setBatchId] = useState(null)
    const [uploadMode, setUploadMode] = useState('files') // 'files' or 'csv'

    // Scroll Terminal to bottom
    const terminalEndRef = useRef(null)
    const scrollToBottom = () => {
        terminalEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
    useEffect(() => {
        scrollToBottom()
    }, [logs])

    const addLog = (msg, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false })
        setLogs(prev => [...prev, { time: timestamp, msg, type }])
    }

    // --- SUBMISSION HANDLERS ---

    // Helper to simulate deep-dive logs during latency
    const [prepStatus, setPrepStatus] = useState({ receptor: 0, ligand: 0, grid: 0 })

    // Helper to simulate granular prep progress
    const simulatePrepProgress = () => {
        // Sequence: Receptor -> Ligand -> Grid
        let r = 0, l = 0, g = 0;

        return setInterval(() => {
            if (r < 100) {
                r += 5;
                setPrepStatus(prev => ({ ...prev, receptor: Math.min(r, 100) }));
            } else if (l < 100) {
                l += 10;
                setPrepStatus(prev => ({ ...prev, ligand: Math.min(l, 100) }));
            } else if (g < 100) {
                g += 20;
                setPrepStatus(prev => ({ ...prev, grid: Math.min(g, 100) }));
            }
        }, 150) // Fast animation
    }

    const handleFilesSubmit = async () => {
        if (!receptorFile || ligandFiles.length === 0) {
            setError('Please upload a receptor and at least one ligand.')
            return
        }

        setLoading(true)
        setProcessingStage('uploading')
        setError(null)
        setLogs([])
        setBatchId(null)

        addLog("Initializing secure batch session...", 'info')

        try {
            // 1. Initialize
            const { data: { session } } = await supabase.auth.getSession()
            const initRes = await fetch(`${API_URL}/jobs/batch/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${session.access_token}` },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filenames: ligandFiles.map(f => f.name)
                })
            })

            if (!initRes.ok) throw new Error('Failed to initialize batch')
            const initData = await initRes.json()
            const newBatchId = initData.batch_id

            addLog(`Batch ID Generated: ${newBatchId}`, 'success')
            addLog(`Secure Upload Channels Opened for ${ligandFiles.length + 1} files`, 'info')

            // 2. Upload Files
            setUploadProgress(10)
            addLog(`Uploading Receptor: ${receptorFile.name}...`, 'info')

            await fetch(initData.upload_urls.receptor_url, { method: 'PUT', body: receptorFile })
            addLog(`Receptor Upload Complete`, 'success')
            setUploadProgress(30)

            addLog(`Uploading ${ligandFiles.length} Ligands...`, 'info')
            // Parallel Uploads
            const startUpload = Date.now()
            let completed = 0
            const total = ligandFiles.length
            const CHUNK_SIZE = 5
            const ligandUrlMap = initData.upload_urls.ligands.reduce((acc, curr) => ({ ...acc, [curr.filename]: curr.url }), {})

            for (let i = 0; i < total; i += CHUNK_SIZE) {
                const chunk = ligandFiles.slice(i, i + CHUNK_SIZE)
                await Promise.all(chunk.map(async file => {
                    const url = ligandUrlMap[file.name]
                    if (url) await fetch(url, { method: 'PUT', body: file })
                    completed++
                    setUploadProgress(30 + Math.round((completed / total) * 60))
                }))
            }
            const duration = ((Date.now() - startUpload) / 1000).toFixed(1)
            addLog(`All Ligands Uploaded in ${duration}s`, 'success')

            // 3. Start Processing
            setProcessingStage('processing')
            addLog("Starting Protein Preparation & Docking Pipeline...", 'warning')

            const listInterval = simulatePrepProgress() // Start fake logs

            const startRes = await fetch(`${API_URL}/jobs/batch/${newBatchId}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    grid_params: { center_x: 0, center_y: 0, center_z: 0, size_x: 20, size_y: 20, size_z: 20 },
                    engine: 'consensus'
                })
            })

            clearInterval(listInterval) // Stop fake logs

            if (!startRes.ok) {
                const errJson = await startRes.json()
                throw new Error(errJson.detail || 'Failed to start batch')
            }

            addLog("Consensus Docking Jobs Submitted Successfully", 'success')
            addLog("AWS Batch Execution: STARTED", 'success')

            setBatchId(newBatchId)
            setProcessingStage('complete')
            setLoading(false)

            trackEvent('batch_docking_submitted', { mode: 'files', count: ligandFiles.length })

        } catch (err) {
            console.error(err)
            setError(err.message)
            addLog(`CRITICAL ERROR: ${err.message}`, 'error')
            setLoading(false)
            setProcessingStage('idle') // Allow retry, but logs stay
        }
    }

    const handleCsvSubmit = async () => {
        if (!csvFile || !receptorFile) {
            setError('Please upload a receptor and a CSV file.')
            return
        }

        setLoading(true)
        setProcessingStage('uploading')
        setError(null)
        setLogs([])
        setBatchId(null)

        addLog("Initializing CSV Batch Session...", 'info')

        try {
            const { data: { session } } = await supabase.auth.getSession()
            const formData = new FormData()
            formData.append('file', csvFile)
            formData.append('receptor', receptorFile)
            formData.append('grid_params', JSON.stringify({
                center_x: 0, center_y: 0, center_z: 0,
                size_x: 20, size_y: 20, size_z: 20
            }))
            formData.append('engine', 'consensus')

            // Simulated upload progress mainly for UX since we send one FormData
            const progressInterval = setInterval(() => {
                setUploadProgress(prev => Math.min(prev + 10, 90))
            }, 500)

            addLog("Uploading and Parsing CSV...", 'info')

            const res = await fetch(`${API_URL}/jobs/batch/submit-csv`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session.access_token}` },
                body: formData
            })

            clearInterval(progressInterval)
            setUploadProgress(100)

            if (!res.ok) {
                const errData = await res.json()
                throw new Error(errData.detail || 'CSV submission failed')
            }

            const data = await res.json()
            addLog(`CSV Parsed: ${data.jobs_created} unique ligands identified`, 'success')
            if (data.conversion_errors > 0) {
                addLog(`Warning: ${data.conversion_errors} SMILES failed conversion`, 'warning')
            }

            setProcessingStage('processing')
            const listInterval = simulatePrepProgress()

            // Artificial delay to let user see the logs if response was too fast
            await new Promise(r => setTimeout(r, 2000))

            clearInterval(listInterval)

            addLog("Batch Submitted to Queue", 'success')
            setBatchId(data.batch_id)
            setProcessingStage('complete')
            setLoading(false)

            trackEvent('batch_docking_submitted', { mode: 'csv', count: data.jobs_created })

        } catch (err) {
            console.error(err)
            setError(err.message)
            addLog(`Error: ${err.message}`, 'error')
            setLoading(false)
            setProcessingStage('idle')
        }
    }


    // --- RENDER HELPERS ---
    const getLogColor = (type) => {
        switch (type) {
            case 'success': return 'text-emerald-400'
            case 'error': return 'text-red-400'
            case 'warning': return 'text-amber-400'
            case 'process': return 'text-indigo-300'
            default: return 'text-slate-300'
        }
    }


    // --- SPLIT VIEW RENDERING (SIDE-BY-SIDE) ---
    if (processingStage !== 'idle' || batchId) {
        return (
            <div className="h-screen flex flex-col md:flex-row bg-slate-900 overflow-hidden">

                {/* LEFT HALF: Context / Summary */}
                <div className="w-full md:w-5/12 h-1/2 md:h-full bg-slate-50 border-r border-slate-700 p-8 overflow-y-auto relative flex-shrink-0">
                    <div className="max-w-xl mx-auto">
                        <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
                            <Activity className="text-indigo-600 animate-pulse" />
                            Active Experiment Context
                        </h2>

                        {/* Vertical Stack for Side Panel */}
                        <div className="flex flex-col gap-4">
                            {/* Card 1: Receptor */}
                            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Receptor Target</div>
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-indigo-50 rounded-lg">
                                        <Database className="w-5 h-5 text-indigo-600" />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="font-bold text-slate-700 truncate">{receptorFile?.name || "Unknown"}</div>
                                        <div className="text-xs text-slate-500">Protein Structure</div>
                                    </div>
                                </div>
                            </div>

                            {/* Card 2: Ligands */}
                            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Ligand Library</div>
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-emerald-50 rounded-lg">
                                        <FlaskConical className="w-5 h-5 text-emerald-600" />
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-700">
                                            {uploadMode === 'files' ? `${ligandFiles.length} Compounds` : csvFile?.name || "CSV Upload"}
                                        </div>
                                        <div className="text-xs text-slate-500">Input Source</div>
                                    </div>
                                </div>
                            </div>

                            {/* Card 3: Engine */}
                            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Docking Engine</div>
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-violet-50 rounded-lg">
                                        <Cpu className="w-5 h-5 text-violet-600" />
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-700">Consensus Mode</div>
                                        <div className="text-xs text-slate-500">Vina + Gnina (AI)</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Progress Bar (Global) */}
                        <div className="mt-8">
                            <div className="flex justify-between text-xs text-slate-500 mb-2">
                                <span>Experiment Progress</span>
                                <span>{processingStage === 'complete' ? '100% - Ready' : `${processingStage === 'uploading' ? Math.round(uploadProgress / 2) : 50 + Math.round(uploadProgress / 2)}%`}</span>
                            </div>
                            <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                                <div
                                    className={`h-full transition-all duration-500 ease-out ${processingStage === 'complete' ? 'bg-emerald-500' : 'bg-indigo-600 query-loading'}`}
                                    style={{ width: processingStage === 'complete' ? '100%' : `${processingStage === 'uploading' ? uploadProgress / 2 : 50 + uploadProgress / 2}%` }}
                                />
                            </div>
                        </div>
                        {/* FINAL BATCH ID DISPLAY */}
                        {processingStage === 'complete' && (
                            <div className="mt-12 bg-white p-6 rounded-2xl shadow-xl border border-indigo-100 flex flex-col items-center animate-fade-in-up">
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Deployed Batch ID</div>
                                <div className="text-3xl font-mono font-bold text-slate-900 bg-slate-50 px-6 py-3 rounded-xl border border-slate-200 select-all mb-6">
                                    {batchId}
                                </div>
                                <button
                                    onClick={() => navigate(`/batch/${batchId}`)}
                                    className="w-full py-4 bg-indigo-600 text-white font-bold rounded-xl hover:bg-indigo-500 transition-all shadow-lg flex items-center justify-center gap-2 transform hover:-translate-y-1"
                                >
                                    <span>View Live Results</span>
                                    <ArrowRight size={20} />
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* RIGHT HALF: Live Deep Dive Console */}
                <div className="w-full md:w-7/12 h-1/2 md:h-full bg-slate-900 border-l border-slate-800 p-8 flex flex-col relative overflow-hidden">
                    {/* Background Pattern */}
                    <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#4f46e5_1px,transparent_1px)] [background-size:16px_16px]"></div>

                    {/* Console Header */}
                    <div className="relative z-10 mb-6 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-slate-800 rounded-lg border border-slate-700">
                                <Terminal className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div>
                                <h3 className="text-slate-100 font-bold font-mono">BioDockify CLI</h3>
                                <div className="flex items-center gap-2 text-xs text-slate-500">
                                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                                    SYSTEM ACTIVE
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="px-3 py-1 bg-slate-800 rounded border border-slate-700 text-xs text-slate-400 font-mono">
                                {logs.length} events
                            </div>
                        </div>
                    </div>

                    {/* Console Content Area */}
                    <div className="flex-1 bg-black/50 rounded-xl border border-slate-700/50 backdrop-blur-sm p-4 overflow-y-auto font-mono text-sm relative custom-scrollbar">
                        {(processingStage === 'processing' || processingStage === 'complete') && (
                            <div className="mb-8">
                                <PreparationProgress currentStep={
                                    processingStage === 'complete' ? 6 :
                                        (prepStatus.grid === 100 ? 5 :
                                            prepStatus.grid > 0 ? 4 :
                                                prepStatus.ligand === 100 ? 3 :
                                                    prepStatus.ligand > 0 ? 2 : 1)
                                } />
                            </div>
                        )}

                        <div className="space-y-2">
                            {logs.map((log, i) => (
                                <div key={i} className={`flex gap-3 animate-fade-in ${getLogColor(log.type)}`}>
                                    <span className="text-slate-600 select-none">[{log.time}]</span>
                                    <span className="break-all">
                                        {log.type === 'success' && '✓ '}
                                        {log.type === 'error' && '✗ '}
                                        {log.type === 'warning' && '⚠ '}
                                        {log.msg}
                                    </span>
                                </div>
                            ))}
                            {processingStage === 'processing' && (
                                <div className="flex gap-3 text-slate-500 animate-pulse">
                                    <span className="text-slate-600">[{new Date().toLocaleTimeString('en-US', { hour12: false })}]</span>
                                    <span>_</span>
                                </div>
                            )}
                            <div ref={terminalEndRef} />
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    // --- DEFAULT INPUT VIEW (When Idle) ---
    return (
        <div className="min-h-screen bg-white">
            <div className="max-w-4xl mx-auto px-4 py-12">

                {/* Header */}
                <div className="mb-12">
                    <Link to="/dashboard" className="text-slate-400 hover:text-slate-600 flex items-center gap-2 mb-4 transition-colors">
                        <ArrowRight className="rotate-180 w-4 h-4" /> Back to Dashboard
                    </Link>
                    <h1 className="text-4xl font-bold text-slate-900 mb-3">New Experiment</h1>
                    <p className="text-slate-500 text-lg">Configure your high-throughput virtual screening campaign.</p>
                </div>

                {/* 1. Receptor Upload */}
                <div className="mb-8">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">1</div>
                        <h2 className="text-lg font-bold text-slate-900">Target Receptor</h2>
                    </div>

                    <div className="border-2 border-dashed border-slate-200 rounded-2xl p-8 hover:bg-slate-50/50 transition-colors cursor-pointer group relative">
                        {!receptorFile ? (
                            <>
                                <input
                                    type="file"
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                    accept=".pdb,.pdbqt,.mol2,.cif,.gro,.prmtop,.psf,.xyz"
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                />
                                <div className="text-center pointer-events-none">
                                    <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                                        <Database className="w-8 h-8 text-indigo-500" />
                                    </div>
                                    <p className="font-bold text-slate-700">Drop Receptor File Here</p>
                                    <p className="text-sm text-slate-400 mt-2">.pdb, .pdbqt, .mol2, .cif (Max 50MB)</p>
                                </div>
                            </>
                        ) : (
                            <div className="flex items-center justify-between bg-indigo-50/50 p-4 rounded-xl border border-indigo-100">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-indigo-100 rounded text-indigo-600">
                                        <FileText size={20} />
                                    </div>
                                    <span className="font-bold text-indigo-900">{receptorFile.name}</span>
                                </div>
                                <button onClick={() => setReceptorFile(null)} className="text-indigo-400 hover:text-indigo-600">✕</button>
                            </div>
                        )}
                    </div>
                </div>

                {/* 2. Ligand Upload */}
                <div className="mb-8">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-4">
                            <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">2</div>
                            <h2 className="text-lg font-bold text-slate-900">Ligand Library</h2>
                        </div>

                        {/* Mode Toggle */}
                        <div className="flex bg-slate-100 p-1 rounded-lg">
                            <button
                                onClick={() => { setUploadMode('files'); setCsvFile(null); }}
                                className={`px-4 py-1.5 text-sm font-bold rounded-md transition-all ${uploadMode === 'files' ? 'bg-white shadow-sm text-indigo-600' : 'text-slate-500'}`}
                            >
                                Multi-File Upload
                            </button>
                            <button
                                onClick={() => { setUploadMode('csv'); setLigandFiles([]); }}
                                className={`px-4 py-1.5 text-sm font-bold rounded-md transition-all ${uploadMode === 'csv' ? 'bg-white shadow-sm text-indigo-600' : 'text-slate-500'}`}
                            >
                                CSV Import
                            </button>
                        </div>
                    </div>

                    <div className="border-2 border-dashed border-slate-200 rounded-2xl p-8 hover:bg-slate-50/50 transition-colors relative min-h-[200px] flex flex-col justify-center">
                        {uploadMode === 'files' ? (
                            ligandFiles.length === 0 ? (
                                <>
                                    <input
                                        type="file"
                                        multiple
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                        accept=".pdbqt,.sdf,.mol2,.mol,.pdb,.xyz"
                                        onChange={(e) => setLigandFiles(Array.from(e.target.files))}
                                    />
                                    <div className="text-center pointer-events-none">
                                        <div className="w-16 h-16 bg-emerald-50 rounded-full flex items-center justify-center mx-auto mb-4">
                                            <Upload className="w-8 h-8 text-emerald-500" />
                                        </div>
                                        <p className="font-bold text-slate-700">Drop Ligand Files Here</p>
                                        <p className="text-sm text-slate-400 mt-2">.pdbqt, .sdf, .mol2 supported</p>
                                    </div>
                                </>
                            ) : (
                                <div>
                                    <div className="flex justify-between items-center mb-4">
                                        <span className="font-bold text-slate-700">{ligandFiles.length} files selected</span>
                                        <button onClick={() => setLigandFiles([])} className="text-red-500 text-sm hover:underline">Clear All</button>
                                    </div>
                                    <div className="max-h-[200px] overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-2">
                                        {ligandFiles.map((f, i) => (
                                            <div key={i} className="flex items-center gap-2 text-sm text-slate-600 bg-slate-50 px-3 py-2 rounded border border-slate-100">
                                                <FileText size={14} className="text-slate-400" />
                                                <span className="truncate">{f.name}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )
                        ) : (
                            // CSV MODE
                            !csvFile ? (
                                <>
                                    <input
                                        type="file"
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                        accept=".csv"
                                        onChange={(e) => setCsvFile(e.target.files[0])}
                                    />
                                    <div className="text-center pointer-events-none">
                                        <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-4">
                                            <FileText className="w-8 h-8 text-blue-500" />
                                        </div>
                                        <p className="font-bold text-slate-700">Drop CSV File Here</p>
                                        <p className="text-sm text-slate-400 mt-2">Must contain 'smiles' column</p>
                                    </div>
                                </>
                            ) : (
                                <div className="flex items-center justify-between bg-blue-50/50 p-4 rounded-xl border border-blue-100">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-blue-100 rounded text-blue-600">
                                            <FileText size={20} />
                                        </div>
                                        <span className="font-bold text-blue-900">{csvFile.name}</span>
                                    </div>
                                    <button onClick={() => setCsvFile(null)} className="text-blue-400 hover:text-blue-600">✕</button>
                                </div>
                            )
                        )}
                    </div>
                </div>

                {/* 3. Engine Selection (Visual Only - Consensus Enforced) */}
                <div className="mb-8">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold">3</div>
                        <h2 className="text-lg font-bold text-slate-900">Processing Engine</h2>
                    </div>

                    <div className="p-6 bg-gradient-to-br from-indigo-50 to-white rounded-2xl border-2 border-indigo-500 shadow-sm flex gap-4">
                        <div className="p-3 bg-white rounded-xl shadow-sm h-fit">
                            <Cpu className="w-8 h-8 text-indigo-600" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2 mb-1">
                                <h3 className="font-bold text-slate-900">BioDockify Consensus Protocol</h3>
                                <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold uppercase rounded-full">Recommended</span>
                            </div>
                            <p className="text-sm text-slate-600 leading-relaxed mb-3">
                                Automatically runs <strong>AutoDock Vina</strong> (Physics-based) and <strong>Gnina</strong> (Deep Learning) in parallel.
                                Results are aggregated to minimize false positives.
                            </p>
                            <div className="flex gap-2">
                                <span className="text-xs px-2 py-1 bg-white border border-slate-200 rounded font-mono text-slate-500">Vina 1.2.3</span>
                                <span className="text-xs px-2 py-1 bg-white border border-slate-200 rounded font-mono text-slate-500">Gnina CNN</span>
                            </div>
                        </div>
                        <div className="ml-auto flex items-center">
                            <CheckCircle2 className="w-6 h-6 text-indigo-600" />
                        </div>
                    </div>
                </div>

                {/* Submit */}
                <div className="flex justify-end pt-8 border-t border-slate-100">
                    {error && (
                        <div className="mr-auto flex items-center gap-2 text-red-600 bg-red-50 px-4 py-2 rounded-lg">
                            <AlertCircle className="w-5 h-5" />
                            <span className="text-sm font-medium">{error}</span>
                        </div>
                    )}

                    <button
                        onClick={uploadMode === 'files' ? handleFilesSubmit : handleCsvSubmit}
                        disabled={loading}
                        className="btn-primary px-8 py-4 text-lg shadow-xl hover:shadow-2xl hover:-translate-y-1 transition-all flex items-center gap-3 disabled:opacity-50 disabled:transform-none"
                    >
                        <span>Start Experiment</span>
                        <Play size={20} fill="currentColor" />
                    </button>
                </div>

            </div>
        </div>
    )
}
