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
            setPrepStatus({ receptor: 100, ligand: 100, grid: 100 })

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


    // --- SPLIT VIEW RENDERING (Status Dashboard) ---
    if (processingStage !== 'idle' || batchId) {
        return (
            <div className="h-screen bg-slate-900 overflow-y-auto">
                <div className="max-w-4xl mx-auto px-6 py-12">
                    <h2 className="text-3xl font-bold text-white mb-8 flex items-center justify-center gap-3">
                        <Activity className="text-indigo-500 animate-pulse w-8 h-8" />
                        <span>Experiment Control Center</span>
                    </h2>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                        {/* Status Cards */}
                        <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 flex items-center gap-4 shadow-lg">
                            <div className="p-3 bg-indigo-500/20 rounded-xl">
                                <Database className="w-6 h-6 text-indigo-400" />
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Target</div>
                                <div className="font-bold text-white text-lg truncate" title={receptorFile?.name}>{receptorFile?.name || "Unknown"}</div>
                            </div>
                        </div>

                        <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 flex items-center gap-4 shadow-lg">
                            <div className="p-3 bg-emerald-500/20 rounded-xl">
                                <FlaskConical className="w-6 h-6 text-emerald-400" />
                            </div>
                            <div className="flex-1">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Compounds</div>
                                <div className="font-bold text-white text-lg">{uploadMode === 'files' ? `${ligandFiles.length}` : "CSV Batch"}</div>
                            </div>
                        </div>

                        <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 flex items-center gap-4 shadow-lg">
                            <div className="p-3 bg-violet-500/20 rounded-xl">
                                <Cpu className="w-6 h-6 text-violet-400" />
                            </div>
                            <div className="flex-1">
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Protocol</div>
                                <div className="font-bold text-white text-lg">Consensus AI</div>
                            </div>
                        </div>
                    </div>

                    {/* Progress Monitor */}
                    <div className="bg-slate-800 p-8 rounded-3xl border border-slate-700 shadow-xl mb-8">
                        <div className="flex justify-between text-sm font-bold text-slate-400 mb-3">
                            <span>Pipeline Velocity</span>
                            <span className="text-indigo-400">{processingStage === 'complete' ? '100% - Ready' : `${processingStage === 'uploading' ? Math.round(uploadProgress / 2) : 50 + Math.round(uploadProgress / 2)}%`}</span>
                        </div>
                        <div className="h-3 w-full bg-slate-700 rounded-full overflow-hidden mb-4 relative">
                            <div
                                className={`h-full transition-all duration-700 ease-out ${processingStage === 'complete' ? 'bg-emerald-500' : 'bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.5)]'}`}
                                style={{ width: processingStage === 'complete' ? '100%' : `${processingStage === 'uploading' ? uploadProgress / 2 : 50 + uploadProgress / 2}%` }}
                            />
                        </div>

                        {/* Preparation Steps Visualization */}
                        {(processingStage === 'processing' || processingStage === 'complete') && (
                            <PreparationProgress
                                currentStep={
                                    processingStage === 'complete' ? 6 :
                                        (prepStatus.grid === 100 ? 5 :
                                            prepStatus.grid > 0 ? 4 :
                                                prepStatus.ligand === 100 ? 3 :
                                                    prepStatus.ligand > 0 ? 2 : 1)
                                }
                                batchId={batchId}
                                isDark={true} // Passing prop for dark mode style
                            />
                        )}
                    </div>

                    {/* Terminal View */}
                    <div className="bg-slate-950 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden font-mono text-sm mb-8">
                        <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex items-center gap-2">
                            <Terminal size={14} className="text-slate-500" />
                            <span className="text-slate-500 text-xs">system_log.txt</span>
                        </div>
                        <div className="p-4 h-48 overflow-y-auto space-y-2">
                            {logs.map((log, i) => (
                                <div key={i} className={`flex gap-3 ${getLogColor(log.type)}`}>
                                    <span className="opacity-50 select-none">[{log.time}]</span>
                                    <span>{log.msg}</span>
                                </div>
                            ))}
                            <div ref={terminalEndRef} />
                        </div>
                    </div>

                    {/* Action Area */}
                    {processingStage === 'complete' && (
                        <div className="flex justify-center animate-fade-in-up pb-12">
                            <button
                                onClick={() => navigate(`/dock/batch/${batchId}`)}
                                className="group relative px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-2xl shadow-2xl hover:shadow-indigo-500/25 transition-all transform hover:-translate-y-1 overflow-hidden"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                                <span className="flex items-center gap-3 text-lg">
                                    Access Batch Results
                                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </span>
                            </button>
                        </div>
                    )}
                </div>
            </div>
        )
    }

    // --- DEFAULT INPUT VIEW (When Idle) ---
    return (
        <div className="min-h-screen bg-slate-50">
            <div className="max-w-5xl mx-auto px-6 py-16">

                {/* Header */}
                <div className="mb-16 text-center">
                    <Link to="/dashboard" className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full text-slate-500 hover:text-indigo-600 text-sm font-medium mb-8 shadow-sm hover:shadow transition-all">
                        <ArrowRight className="rotate-180 w-4 h-4" /> Back to Dashboard
                    </Link>
                    <h1 className="text-5xl font-bold text-slate-900 mb-6 bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-700">
                        Batch Docking Page
                    </h1>
                    <p className="text-slate-500 text-xl max-w-2xl mx-auto leading-relaxed">
                        Deploy massive virtual screening campaigns using our consensus docking engine.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                    {/* 1. Receptor Upload */}
                    <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-100 hover:shadow-xl hover:border-indigo-100 transition-all duration-300">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-xl">1</div>
                            <div>
                                <h2 className="text-xl font-bold text-slate-900">Target Receptor</h2>
                                <p className="text-slate-500 text-sm">Protein structure (.pdb, .pdbqt)</p>
                            </div>
                        </div>

                        <div className="border-2 border-dashed border-slate-200 rounded-2xl p-8 hover:bg-slate-50 transition-colors cursor-pointer group relative h-48 flex flex-col items-center justify-center text-center">
                            {!receptorFile ? (
                                <>
                                    <input
                                        type="file"
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                        accept=".pdb,.pdbqt,.mol2,.cif,.gro,.prmtop,.psf,.xyz"
                                        onChange={(e) => setReceptorFile(e.target.files[0])}
                                    />
                                    <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                        <Database className="w-8 h-8 text-indigo-500" />
                                    </div>
                                    <p className="font-bold text-slate-700 text-lg">Upload Receptor</p>
                                    <p className="text-sm text-slate-400 mt-2">Max 50MB</p>
                                </>
                            ) : (
                                <div className="w-full">
                                    <div className="flex items-center justify-between bg-indigo-50 p-4 rounded-xl border border-indigo-100 mb-4">
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <div className="p-2 bg-white rounded text-indigo-600 shadow-sm">
                                                <FileText size={20} />
                                            </div>
                                            <span className="font-bold text-indigo-900 truncate">{receptorFile.name}</span>
                                        </div>
                                    </div>
                                    <button onClick={() => setReceptorFile(null)} className="text-sm text-red-500 font-medium hover:underline">Remove file</button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* 2. Ligand Upload */}
                    <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-100 hover:shadow-xl hover:border-emerald-100 transition-all duration-300">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center font-bold text-xl">2</div>
                                <div>
                                    <h2 className="text-xl font-bold text-slate-900">Ligand Library</h2>
                                    <p className="text-slate-500 text-sm">Small molecules</p>
                                </div>
                            </div>

                            {/* Toggle */}
                            <div className="flex bg-slate-100 p-1 rounded-xl">
                                <button
                                    onClick={() => { setUploadMode('files'); setCsvFile(null); }}
                                    className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${uploadMode === 'files' ? 'bg-white shadow text-slate-900' : 'text-slate-500'}`}
                                >
                                    Files
                                </button>
                                <button
                                    onClick={() => { setUploadMode('csv'); setLigandFiles([]); }}
                                    className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${uploadMode === 'csv' ? 'bg-white shadow text-slate-900' : 'text-slate-500'}`}
                                >
                                    CSV
                                </button>
                            </div>
                        </div>

                        <div className="border-2 border-dashed border-slate-200 rounded-2xl p-8 hover:bg-slate-50 transition-colors relative h-48 flex flex-col items-center justify-center text-center">
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
                                        <div className="w-16 h-16 bg-emerald-50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                            <Upload className="w-8 h-8 text-emerald-500" />
                                        </div>
                                        <p className="font-bold text-slate-700 text-lg">Upload Ligands</p>
                                        <p className="text-sm text-slate-400 mt-2">.pdbqt, .sdf, .mol2</p>
                                    </>
                                ) : (
                                    <div className="w-full h-full flex flex-col">
                                        <div className="flex justify-between items-center mb-4">
                                            <span className="font-bold text-slate-700">{ligandFiles.length} files prepared</span>
                                            <button onClick={() => setLigandFiles([])} className="text-red-500 text-xs font-bold bg-red-50 px-2 py-1 rounded hover:bg-red-100">CLEAR</button>
                                        </div>
                                        <div className="flex-1 overflow-y-auto grid grid-cols-1 gap-2 text-left pr-2 custom-scrollbar">
                                            {ligandFiles.slice(0, 10).map((f, i) => ( // Show first 10 preview
                                                <div key={i} className="flex items-center gap-2 text-xs text-slate-600 bg-slate-50 px-3 py-2 rounded-lg border border-slate-100">
                                                    <FileText size={12} className="text-slate-400 shrink-0" />
                                                    <span className="truncate">{f.name}</span>
                                                </div>
                                            ))}
                                            {ligandFiles.length > 10 && (
                                                <div className="text-center text-xs text-slate-400 py-1">
                                                    + {ligandFiles.length - 10} more files
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )
                            ) : (
                                !csvFile ? (
                                    <>
                                        <input
                                            type="file"
                                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                            accept=".csv"
                                            onChange={(e) => setCsvFile(e.target.files[0])}
                                        />
                                        <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-4">
                                            <FileText className="w-8 h-8 text-blue-500" />
                                        </div>
                                        <p className="font-bold text-slate-700 text-lg">Upload CSV</p>
                                        <p className="text-sm text-slate-400 mt-2">Required column: 'smiles'</p>
                                    </>
                                ) : (
                                    <div className="w-full">
                                        <div className="flex items-center justify-between bg-blue-50 p-4 rounded-xl border border-blue-100 mb-4">
                                            <div className="flex items-center gap-3 overflow-hidden">
                                                <div className="p-2 bg-white rounded text-blue-600 shadow-sm">
                                                    <FileText size={20} />
                                                </div>
                                                <span className="font-bold text-blue-900 truncate">{csvFile.name}</span>
                                            </div>
                                        </div>
                                        <button onClick={() => setCsvFile(null)} className="text-sm text-red-500 font-medium hover:underline">Remove file</button>
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                </div>

                {/* 3. Engine Selection */}
                <div className="bg-gradient-to-br from-indigo-900 to-indigo-800 rounded-3xl p-8 text-white shadow-xl relative overflow-hidden mb-12">
                    <div className="absolute top-0 right-0 p-32 bg-white/5 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>

                    <div className="flex items-start gap-6 relative z-10">
                        <div className="w-12 h-12 rounded-2xl bg-white/10 text-white flex items-center justify-center font-bold text-xl backdrop-blur-sm border border-white/20">3</div>
                        <div className="flex-1">
                            <div className="flex flex-col md:flex-row md:items-center gap-4 mb-4">
                                <h2 className="text-2xl font-bold">BioDockify Consensus Protocol</h2>
                                <span className="px-3 py-1 bg-white/20 backdrop-blur rounded-full text-xs font-bold uppercase tracking-wide border border-white/30">
                                    Industry Standard
                                </span>
                            </div>
                            <p className="text-indigo-100 text-lg mb-6 max-w-2xl">
                                Combines the physics-based accuracy of <strong className="text-white">AutoDock Vina</strong> with the deep learning capabilities of <strong className="text-white">Gnina</strong> for superior hit enrichment.
                            </p>
                            <div className="flex gap-4">
                                <div className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg border border-white/10">
                                    <Cpu size={16} />
                                    <span className="font-mono text-sm">Vina 1.2.3</span>
                                </div>
                                <div className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg border border-white/10">
                                    <Activity size={16} />
                                    <span className="font-mono text-sm">Gnina CNN</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Submit Bar */}
                <div className="flex justify-end pt-8">
                    {error && (
                        <div className="mr-4 flex items-center gap-3 text-red-600 bg-red-50 px-6 py-4 rounded-xl border border-red-100 animate-pulse">
                            <AlertCircle className="w-6 h-6" />
                            <span className="font-medium">{error}</span>
                        </div>
                    )}
                    <button
                        onClick={uploadMode === 'files' ? handleFilesSubmit : handleCsvSubmit}
                        disabled={loading}
                        className="btn-primary pl-10 pr-8 py-5 text-xl rounded-2xl shadow-xl hover:shadow-2xl hover:-translate-y-1 transition-all flex items-center gap-4 disabled:opacity-50 disabled:transform-none bg-slate-900 text-white hover:bg-slate-800"
                    >
                        <span>Start Experiment</span>
                        <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                            <Play size={14} fill="currentColor" />
                        </div>
                    </button>
                </div>

            </div>
        </div>
    )
}
