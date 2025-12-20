import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import PreparationProgress from '../components/PreparationProgress'
import { Upload, FileText, Database, Cpu, Play, CheckCircle2, AlertCircle, ArrowRight, FlaskConical } from 'lucide-react'

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
                    headers: { 'Authorization': `Bearer ${session.access_token}` },
                    body: formData
                })
                setUploadProgress(80)
                if (!res.ok) {
                    const err = await res.json()
                    throw new Error(err.detail || 'CSV batch submission failed')
                }
                const result = await res.json()
                setUploadProgress(100)
                // alert(`Batch Processing Started!\n✅ ${result.jobs_created} jobs running.`)
                navigate(`/dock/batch/${result.batch_id}`)

            } else {
                // FILES FLOW
                const createRes = await fetch(`${API_URL}/jobs/batch/submit`, {
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

                if (!createRes.ok) {
                    const err = await createRes.json()
                    throw new Error(err.detail || 'Failed to create batch')
                }

                const { batch_id, upload_urls } = await createRes.json()
                setBatchId(batch_id)

                // Upload Receptor
                await uploadFile(upload_urls.receptor_url, receptorFile)
                setUploadProgress(10)

                // Upload Ligands
                const totalLigands = ligandFiles.length
                let uploadedCount = 0
                const urlMap = {}
                upload_urls.ligands.forEach(l => urlMap[l.filename] = l.url)

                for (const file of ligandFiles) {
                    const url = urlMap[file.name]
                    if (url) {
                        await uploadFile(url, file)
                        uploadedCount++
                        setUploadProgress(10 + Math.floor((uploadedCount / totalLigands) * 80))
                    }
                }

                // Simulate/Show Prep Steps
                const steps = ['Protein Prepared', 'Water Removal', 'Ligand Prepared', 'Config Generated', 'Grid Ready']
                for (let i = 0; i < steps.length; i++) {
                    setPreparationStep(i + 1)
                    await new Promise(r => setTimeout(r, 800))
                }

                const startRes = await fetch(`${API_URL}/jobs/batch/${batch_id}/start`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${session.access_token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        grid_params: {
                            grid_center_x: parseFloat(gridParams.center_x),
                            grid_center_y: parseFloat(gridParams.center_y),
                            grid_center_z: parseFloat(gridParams.center_z),
                            grid_size_x: parseFloat(gridParams.size_x),
                            grid_size_y: parseFloat(gridParams.size_y),
                            grid_size_z: parseFloat(gridParams.size_z)
                        },
                        engine: engine
                    })
                })

                if (!startRes.ok) {
                    const text = await startRes.text()
                    throw new Error(`Server Error: ${text.substring(0, 100)}`)
                }

                setUploadProgress(100)
                setLoading(false)
                navigate(`/dock/batch/${batch_id}`)
            }

        } catch (err) {
            console.error(err)
            setError(err.message)
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* HERO SECTION */}
            <div className="bg-gradient-to-br from-indigo-900 via-indigo-800 to-purple-900 pt-32 pb-20 text-white relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('/assets/images/grid.svg')] opacity-10"></div>
                <div className="container mx-auto px-4 relative z-10 text-center">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-indigo-100 text-sm font-semibold mb-6 backdrop-blur-sm">
                        <FlaskConical className="w-4 h-4" />
                        <span>High-Throughput Virtual Screening</span>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">
                        Launch Your Docking Campaign
                    </h1>
                    <p className="text-indigo-200 text-lg max-w-2xl mx-auto">
                        Screen thousands of compounds against your target protein using our consensus AI scoring engine.
                        Drag, drop, and discover.
                    </p>
                </div>
            </div>

            <main className="container mx-auto px-4 -mt-10 relative z-20">
                <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden">

                    {/* PROGRESS HEADER */}
                    <div className="bg-slate-50 border-b border-slate-200 px-8 py-4">
                        <div className="flex items-center justify-between text-sm font-medium text-slate-500">
                            <div className="flex items-center gap-2 text-primary-600">
                                <span className="w-6 h-6 rounded-full bg-primary-100 flex items-center justify-center text-xs font-bold">1</span>
                                Target
                            </div>
                            <ArrowRight className="w-4 h-4 text-slate-300" />
                            <div className={`flex items-center gap-2 ${receptorFile ? 'text-primary-600' : ''}`}>
                                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${receptorFile ? 'bg-primary-100' : 'bg-slate-200'}`}>2</span>
                                Ligands
                            </div>
                            <ArrowRight className="w-4 h-4 text-slate-300" />
                            <div className={`flex items-center gap-2 ${ligandFiles.length > 0 || csvFile ? 'text-primary-600' : ''}`}>
                                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${ligandFiles.length > 0 || csvFile ? 'bg-primary-100' : 'bg-slate-200'}`}>3</span>
                                Launch
                            </div>
                        </div>
                    </div>

                    <form onSubmit={uploadMode === 'csv' ? handleCSVSubmit : handleSubmit} className="p-8 space-y-8">

                        {/* 1. RECEPTOR UPLOAD */}
                        <section>
                            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Database className="w-5 h-5 text-indigo-500" /> Target Protein
                            </h3>
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${receptorFile ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}>
                                <input
                                    type="file"
                                    id="receptor-upload"
                                    accept=".pdb,.pdbqt"
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="hidden"
                                />
                                <label htmlFor="receptor-upload" className="cursor-pointer block">
                                    {receptorFile ? (
                                        <div className="text-indigo-700">
                                            <CheckCircle2 className="w-12 h-12 mx-auto mb-2 text-green-500" />
                                            <div className="font-bold text-lg">{receptorFile.name}</div>
                                            <div className="text-sm opacity-75">Click to change file</div>
                                        </div>
                                    ) : (
                                        <div>
                                            <Upload className="w-12 h-12 mx-auto mb-2 text-slate-400" />
                                            <span className="block font-semibold text-slate-700">Drop PDB/PDBQT file here</span>
                                            <span className="text-sm text-slate-500">or click to browse</span>
                                        </div>
                                    )}
                                </label>
                            </div>
                            <div className="flex gap-4 mt-2 px-1">
                                <div className="flex items-center gap-2 text-sm text-slate-600">
                                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Auto-Remove solvent
                                </div>
                                <div className="flex items-center gap-2 text-sm text-slate-600">
                                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Add polar hydrogens
                                </div>
                            </div>
                        </section>

                        <hr className="border-slate-100" />

                        {/* 2. LIGAND UPLOAD */}
                        <section>
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                    <FileText className="w-5 h-5 text-emerald-500" /> Ligand Library
                                </h3>
                                <div className="flex bg-slate-100 p-1 rounded-lg text-sm">
                                    <button
                                        type="button"
                                        onClick={() => setUploadMode('files')}
                                        className={`px-3 py-1 rounded-md transition-all ${uploadMode === 'files' ? 'bg-white shadow text-slate-900 font-medium' : 'text-slate-500 hover:text-slate-700'}`}
                                    >
                                        Files
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setUploadMode('csv')}
                                        className={`px-3 py-1 rounded-md transition-all ${uploadMode === 'csv' ? 'bg-white shadow text-slate-900 font-medium' : 'text-slate-500 hover:text-slate-700'}`}
                                    >
                                        SMILES (CSV)
                                    </button>
                                </div>
                            </div>

                            {uploadMode === 'files' ? (
                                <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${ligandFiles.length > 0 ? 'border-emerald-500 bg-emerald-50' : 'border-slate-300 hover:border-emerald-400 hover:bg-slate-50'}`}>
                                    <input
                                        type="file"
                                        id="ligand-upload"
                                        multiple
                                        accept=".pdbqt,.sdf,.mol2"
                                        onChange={handleLigandChange}
                                        className="hidden"
                                    />
                                    <label htmlFor="ligand-upload" className="cursor-pointer block">
                                        {ligandFiles.length > 0 ? (
                                            <div className="text-emerald-700">
                                                <div className="text-3xl font-bold mb-1">{ligandFiles.length}</div>
                                                <div className="font-medium">Files Selected</div>
                                                <div className="text-sm opacity-75 mt-1">Click to add more</div>
                                            </div>
                                        ) : (
                                            <div>
                                                <Database className="w-12 h-12 mx-auto mb-2 text-slate-400" />
                                                <span className="block font-semibold text-slate-700">Drop SDF/PDBQT/MOL2 files</span>
                                                <span className="text-sm text-slate-500">Up to 100 files supported</span>
                                            </div>
                                        )}
                                    </label>
                                </div>
                            ) : (
                                <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${csvFile ? 'border-emerald-500 bg-emerald-50' : 'border-slate-300 hover:border-emerald-400 hover:bg-slate-50'}`}>
                                    <input
                                        type="file"
                                        id="csv-upload"
                                        accept=".csv"
                                        onChange={(e) => setCsvFile(e.target.files[0])}
                                        className="hidden"
                                    />
                                    <label htmlFor="csv-upload" className="cursor-pointer block">
                                        {csvFile ? (
                                            <div className="text-emerald-700">
                                                <FileText className="w-12 h-12 mx-auto mb-2 text-emerald-500" />
                                                <div className="font-bold">{csvFile.name}</div>
                                                <div className="text-sm opacity-75">Click to change</div>
                                            </div>
                                        ) : (
                                            <div>
                                                <FileText className="w-12 h-12 mx-auto mb-2 text-slate-400" />
                                                <span className="block font-semibold text-slate-700">Upload CSV with 'smiles' column</span>
                                            </div>
                                        )}
                                    </label>
                                </div>
                            )}
                        </section>

                        <hr className="border-slate-100" />

                        {/* 3. ENGINE SELECTION */}
                        <section>
                            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Cpu className="w-5 h-5 text-purple-500" /> Docking Engine
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <button
                                    type="button"
                                    onClick={() => setEngine('consensus')}
                                    className={`relative p-4 rounded-xl border-2 text-left transition-all ${engine === 'consensus' ? 'border-purple-500 bg-purple-50' : 'border-slate-200 hover:border-purple-200'}`}
                                >
                                    {engine === 'consensus' && <div className="absolute top-2 right-2 text-purple-600"><CheckCircle2 className="w-5 h-5" /></div>}
                                    <div className="font-bold text-slate-900 mb-1">Consensus (Recommended)</div>
                                    <div className="text-sm text-slate-500">Runs Vina + Gnina (CNN) together for highest accuracy.</div>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setEngine('vina')}
                                    className={`relative p-4 rounded-xl border-2 text-left transition-all ${engine === 'vina' ? 'border-purple-500 bg-purple-50' : 'border-slate-200 hover:border-purple-200'}`}
                                >
                                    {engine === 'vina' && <div className="absolute top-2 right-2 text-purple-600"><CheckCircle2 className="w-5 h-5" /></div>}
                                    <div className="font-bold text-slate-900 mb-1">AutoDock Vina</div>
                                    <div className="text-sm text-slate-500">Standard for speed and reliability.</div>
                                </button>
                            </div>
                        </section>

                        {/* SUBMIT BUTTON */}
                        <div className="pt-4">
                            {error && (
                                <div className="mb-4 p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-3 border border-red-100">
                                    <AlertCircle className="w-5 h-5" /> {error}
                                </div>
                            )}

                            {(loading || preparationStep > 0) ? (
                                <div className="space-y-4">
                                    <PreparationProgress currentStep={preparationStep || 1} />
                                    <button disabled className="w-full py-4 rounded-xl font-bold text-lg bg-slate-100 text-slate-400 cursor-not-allowed">
                                        Processing...
                                    </button>
                                </div>
                            ) : (
                                <button
                                    type="submit"
                                    className="w-full py-4 rounded-xl font-bold text-lg text-white shadow-xl shadow-indigo-200 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 transition-all flex items-center justify-center gap-2 group"
                                >
                                    <Play className="w-5 h-5 fill-current group-hover:scale-110 transition-transform" />
                                    Launch Virtual Screening
                                </button>
                            )}
                        </div>
                    </form>
                </div>
            </main>
        </div>
    )
}
