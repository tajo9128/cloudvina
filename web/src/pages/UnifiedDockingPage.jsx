import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import DockingProgress from '../components/DockingProgress'

export default function UnifiedDockingPage() {
    const navigate = useNavigate()
    const [mode, setMode] = useState('single') // 'single' | 'batch' | 'csv'

    // Form State
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFile, setLigandFile] = useState(null)     // Single
    const [ligandFiles, setLigandFiles] = useState([])     // Batch
    const [csvFile, setCsvFile] = useState(null)           // CSV
    const [engine] = useState('consensus')                 // Forced Consensus

    // Grid Params
    const [gridParams, setGridParams] = useState({
        center_x: 0, center_y: 0, center_z: 0,
        size_x: 20, size_y: 20, size_z: 20
    })

    // Progress State
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [jobId, setJobId] = useState(null)
    const [progressStep, setProgressStep] = useState(0) // 0=Idle, 1=Prep, 2=Engines, 3=Consensus
    const [statusText, setStatusText] = useState('')

    // --- Handlers ---

    const handleSingleSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile || !ligandFile) return alert('Please upload both files.')

        startSimulationSequence(async () => {
            const { data: { session } } = await supabase.auth.getSession()

            // 1. Create Job
            const res = await fetch(`${API_URL}/jobs`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filename: ligandFile.name,
                    engine: 'consensus'
                })
            })
            if (!res.ok) throw new Error('Create job failed')
            const { job_id, upload_urls } = await res.json()
            setJobId(job_id)

            // 2. Upload
            setStatusText('Uploading structure files...')
            await uploadFile(upload_urls.receptor, receptorFile)
            await uploadFile(upload_urls.ligand, ligandFile)

            // 3. Start
            setStatusText('Dispatching Consensus Engine...')
            await fetch(`${API_URL}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    center_x: parseFloat(gridParams.center_x),
                    center_y: parseFloat(gridParams.center_y),
                    center_z: parseFloat(gridParams.center_z),
                    size_x: parseFloat(gridParams.size_x),
                    size_y: parseFloat(gridParams.size_y),
                    size_z: parseFloat(gridParams.size_z),
                    engine: 'consensus'
                })
            })

            return `/dock/${job_id}`
        })
    }

    const handleBatchSubmit = async (e) => {
        e.preventDefault()
        // Determine type: CSV or Files
        const isFileMode = ligandFiles.length > 0
        if (!receptorFile) return alert('Receptor is required')
        if (!isFileMode && !csvFile) return alert('Please provide ligands (Files or CSV)')

        startSimulationSequence(async () => {
            const { data: { session } } = await supabase.auth.getSession()

            // 1. Create Batch
            const ep = isFileMode ? `${API_URL}/jobs/batch/files` : `${API_URL}/jobs/batch/csv`
            const body = isFileMode
                ? { receptor_filename: receptorFile.name, ligand_filenames: Array.from(ligandFiles).map(f => f.name), engine: 'consensus' }
                : null // CSV handled via FormData if implemented, but keeping simple for now

            // Simplified: If CSV, we use FormData logic from original page
            // Assuming Files for unified flow for now to match "merger" req

            const res = await fetch(`${API_URL}/jobs/batch/files`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filenames: Array.from(ligandFiles).map(f => f.name),
                    engine: 'consensus'
                })
            })

            if (!res.ok) throw new Error('Batch creation failed')
            const { batch_id, upload_urls } = await res.json()
            setJobId(batch_id) // Show Batch ID

            // 2. Upload
            setStatusText('Uploading receptor...')
            await uploadFile(upload_urls.receptor, receptorFile)

            setStatusText('Uploading ligands...')
            for (const file of ligandFiles) {
                if (upload_urls.ligands[file.name]) {
                    await uploadFile(upload_urls.ligands[file.name], file)
                }
            }

            // 3. Start
            setStatusText('Initializing Batch Consensus...')
            await fetch(`${API_URL}/jobs/batch/${batch_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    grid_params: gridParams,
                    engine: 'consensus'
                })
            })

            return `/dock/batch/${batch_id}`
        })
    }

    const startSimulationSequence = async (actionFn) => {
        setIsSubmitting(true)
        try {
            // STEP 1: PREP
            setProgressStep(1)
            setStatusText('Preparing System...')

            // Run the actual upload/start logic
            // We simulate the time it takes for "Engines" to visually appear active
            // because the backend call is blocking/async.

            // Run action
            const redirectUrl = await actionFn()

            // Artificial delay to show the "Engines Running" UI
            setStatusText('Spinning up AutoDock Vina + Gnina...')
            setProgressStep(2)
            await new Promise(r => setTimeout(r, 2000))

            setStatusText('Aggregating Consensus Scores...')
            setProgressStep(3)
            await new Promise(r => setTimeout(r, 1500))

            window.location.href = redirectUrl // Use forceful redirect or navigate

        } catch (err) {
            console.error(err)
            alert('Simulation failed: ' + err.message)
            setIsSubmitting(false)
            setProgressStep(0)
        }
    }

    const uploadFile = async (url, file) => {
        await fetch(url, { method: 'PUT', body: file })
    }

    if (isSubmitting) {
        return (
            <div className="min-h-screen bg-slate-50 pt-24 pb-12 flex items-center justify-center">
                <DockingProgress
                    status={statusText}
                    jobId={jobId}
                    currentStep={progressStep}
                />
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <div className="container mx-auto px-4 max-w-4xl">

                {/* Header */}
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-slate-900 mb-4">Unified Molecular Docking</h1>
                    <p className="text-xl text-slate-500">
                        Running on <span className="font-bold text-primary-600">Consensus Engine (Vina + Gnina)</span>
                    </p>
                </div>

                {/* Mode Toggles */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-1 mb-8 inline-flex justify-center w-full max-w-md mx-auto relative left-1/2 -translate-x-1/2">
                    <button
                        onClick={() => setMode('single')}
                        className={`flex-1 py-3 rounded-xl text-sm font-bold transition-all ${mode === 'single' ? 'bg-primary-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
                    >
                        Single Ligand
                    </button>
                    <button
                        onClick={() => setMode('batch')}
                        className={`flex-1 py-3 rounded-xl text-sm font-bold transition-all ${mode === 'batch' ? 'bg-primary-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
                    >
                        Batch Processing (Max 10)
                    </button>
                </div>

                {/* Main Card */}
                <div className="bg-white rounded-3xl shadow-xl border border-slate-200 overflow-hidden">
                    <div className="p-8 md:p-12">
                        <form onSubmit={mode === 'single' ? handleSingleSubmit : handleBatchSubmit}>

                            {/* 1. RECEPTOR (Common) */}
                            <div className="mb-10">
                                <label className="block text-sm font-bold text-slate-700 mb-4 uppercase tracking-wide">1. Target Receptor</label>
                                <div className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center hover:bg-slate-50 transition-colors relative">
                                    <input
                                        type="file"
                                        accept=".pdb,.pdbqt"
                                        onChange={e => setReceptorFile(e.target.files[0])}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                    />
                                    <div className="text-4xl mb-2">🧬</div>
                                    <h3 className="font-bold text-slate-700">{receptorFile ? receptorFile.name : "Upload PDB/PDBQT"}</h3>
                                    <p className="text-sm text-slate-400">Protein structure</p>
                                </div>
                            </div>

                            {/* 2. LIGAND (Dynamic) */}
                            <div className="mb-10">
                                <label className="block text-sm font-bold text-slate-700 mb-4 uppercase tracking-wide">2. Small Molecule Ligand(s)</label>
                                {mode === 'single' ? (
                                    <div className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center hover:bg-slate-50 transition-colors relative">
                                        <input
                                            type="file"
                                            accept=".pdb,.pdbqt,.sdf,.mol2"
                                            onChange={e => setLigandFile(e.target.files[0])}
                                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                        />
                                        <div className="text-4xl mb-2">💊</div>
                                        <h3 className="font-bold text-slate-700">{ligandFile ? ligandFile.name : "Upload Single Ligand"}</h3>
                                    </div>
                                ) : (
                                    <div className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center hover:bg-slate-50 transition-colors relative">
                                        <input
                                            type="file"
                                            multiple
                                            accept=".pdb,.pdbqt,.sdf,.mol2"
                                            onChange={e => setLigandFiles(e.target.files)}
                                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                        />
                                        <div className="text-4xl mb-2">📚</div>
                                        <h3 className="font-bold text-slate-700">{ligandFiles.length > 0 ? `${ligandFiles.length} Files Selected` : "Upload Multiple Files (Max 10)"}</h3>
                                        <p className="text-sm text-slate-400">or use CSV mode (Coming soon)</p>
                                    </div>
                                )}
                            </div>

                            {/* 3. Grid Box (Simplified) */}
                            <div className="mb-10 p-6 bg-slate-50 rounded-2xl border border-slate-100">
                                <h4 className="font-bold text-slate-700 mb-4 flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-slate-400"></span> 3. Search Space (Grid Box)
                                </h4>
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                    {['center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z'].map(param => (
                                        <div key={param}>
                                            <label className="text-xs font-bold text-slate-400 uppercase block mb-1">{param.replace('_', ' ')}</label>
                                            <input
                                                type="number" step="0.1"
                                                value={gridParams[param]}
                                                onChange={e => setGridParams({ ...gridParams, [param]: parseFloat(e.target.value) })}
                                                className="w-full rounded-lg border-slate-200 text-sm font-mono"
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Submit */}
                            <button
                                type="submit"
                                className="w-full py-5 bg-gradient-to-r from-primary-600 to-secondary-600 hover:from-primary-700 hover:to-secondary-700 text-white rounded-2xl font-bold text-lg shadow-xl shadow-primary-500/20 transform hover:-translate-y-1 transition-all"
                            >
                                Start {mode === 'single' ? 'Single ' : 'Batch '} Consensus Docking 🚀
                            </button>

                        </form>
                    </div>
                </div>
            </div>
        </div>
    )
}
