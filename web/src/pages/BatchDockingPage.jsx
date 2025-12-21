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
    const [error, setError] = useState(null)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [batchId, setBatchId] = useState(null)
    const [preparationStep, setPreparationStep] = useState(0)

    // Upload Mode Toggle
    const [uploadMode, setUploadMode] = useState('files') // 'files' or 'csv'
    const [csvFile, setCsvFile] = useState(null)
    const [engine, setEngine] = useState('consensus')

    // Grid Box State
    const [gridParams, setGridParams] = useState({
        center_x: 0, center_y: 0, center_z: 0,
        size_x: 20, size_y: 20, size_z: 20
    })

    const handleLigandChange = (e) => {
        if (e.target.files) {
            const files = Array.from(e.target.files)
            if (files.length > 100) {
                setError("Maximum 100 files allowed per job.")
                return
            }
            setLigandFiles(files)
            setError(null)
        }
    }

    const uploadFile = async (url, file) => {
        const res = await fetch(url, { method: 'PUT', body: file })
        if (!res.ok) throw new Error(`Failed to upload ${file.name}`)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile || ligandFiles.length === 0) {
            setError('Please select a receptor and at least one ligand.')
            return
        }
        await processSubmission()
    }

    const handleCSVSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile) {
            setError('❌ Missing Receptor file.')
            return
        }
        if (!csvFile) {
            setError('❌ Missing CSV file.')
            return
        }
        await processSubmission(true)
    }

    const processSubmission = async (isCsv = false) => {
        setLoading(true)
        setError(null)
        setUploadProgress(0)

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // Track Job Submission
            trackEvent('job:submitting', {
                upload_mode: isCsv ? 'csv' : 'files',
                receptor_name: receptorFile?.name,
                ligand_count: isCsv ? null : ligandFiles.length, // CSV count unknown until processed
                engine: engine,
                grid_params: gridParams
            });

            if (isCsv) {
                // CSV FLOW
                setUploadProgress(20)
                const formData = new FormData()
                formData.append('receptor_file', receptorFile)
                formData.append('csv_file', csvFile)
                Object.keys(gridParams).forEach(k => formData.append(`grid_${k}`, gridParams[k]))
                formData.append('engine', engine)

                const res = await fetch(`${API_URL}/jobs/batch/submit-csv`, {
                    method: 'POST',
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
                                    className="w-full py-4 rounded-xl font-bold text-lg text-white shadow-xl shadow-indigo-200 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 transition-all flex items-center justify-center gap-2 group"
                                >
                                <Play className="w-5 h-5 fill-current group-hover:scale-110 transition-transform" />
                                Launch Virtual Screening
                            </button>
                            )}
                        </div>
                    </form>
                </div>
            </main >
        </div >
    )
}
