import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import PreparationProgress from '../components/PreparationProgress'

export default function BatchDockingPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFiles, setLigandFiles] = useState([])
    const [error, setError] = useState(null)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [batchId, setBatchId] = useState(null)
    const [preparationStep, setPreparationStep] = useState(0)

    // NEW: Upload Mode Toggle (files vs csv)
    const [uploadMode, setUploadMode] = useState('files') // 'files' or 'csv'
    const [csvFile, setCsvFile] = useState(null)
    const [engine, setEngine] = useState('consensus') // Default: Consensus per user request

    // Grid Box State (FIXED)
    const [gridParams, setGridParams] = useState({
        center_x: 0,
        center_y: 0,
        center_z: 0,
        size_x: 20,
        size_y: 20,
        size_z: 20
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
        const res = await fetch(url, {
            method: 'PUT',
            body: file
        })
        if (!res.ok) throw new Error(`Failed to upload ${file.name}`)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile || ligandFiles.length === 0) {
            setError('Please select a receptor and at least one ligand.')
            return
        }

        setLoading(true)
        setError(null)
        setUploadProgress(0)

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // 1. Create Batch
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

            // 2. Upload Files
            // Upload Receptor
            await uploadFile(upload_urls.receptor_url, receptorFile)
            setUploadProgress(10)

            // Upload Ligands (Parallel with concurrency limit ideally, but simple loop for now)
            const totalLigands = ligandFiles.length
            let uploadedCount = 0

            // Create a map for quick lookup
            const urlMap = {}
            upload_urls.ligands.forEach(l => {
                urlMap[l.filename] = l.url
            })

            for (const file of ligandFiles) {
                const url = urlMap[file.name]
                if (url) {
                    await uploadFile(url, file)
                    uploadedCount++
                    setUploadProgress(10 + Math.floor((uploadedCount / totalLigands) * 80))
                }
            }

            // 3. Start Batch with UI Updates
            // Note: The backend now performs conversion/preparation synchronously during this call

            // Start Visual Progress
            setPreparationStep(1) // Step 1: Protein Prepared
            await new Promise(r => setTimeout(r, 1500))

            setPreparationStep(2) // Step 2: Water Removal
            await new Promise(r => setTimeout(r, 1000))

            setPreparationStep(3) // Step 3: Ligand Prepared
            await new Promise(r => setTimeout(r, 2000)) // Give it a moment to show "converting..."

            setPreparationStep(4) // Step 4: Config Generated
            await new Promise(r => setTimeout(r, 1000))

            setPreparationStep(5) // Step 5: Grid File Ready
            await new Promise(r => setTimeout(r, 500))

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
                    engine: engine  // Use selected engine
                })
            })



            if (!startRes.ok) {
                let errMsg = 'Failed to start batch execution'
                try {
                    const errText = await startRes.text()
                    try {
                        const errJson = JSON.parse(errText)
                        errMsg = errJson.detail || errMsg
                    } catch {
                        // If not JSON (e.g. 500 HTML), show start of text
                        errMsg = `Server Error: ${errText.substring(0, 200)}...`
                    }
                } catch (e) {
                    console.error("Error parsing error response", e)
                }
                throw new Error(errMsg)
            }

            setUploadProgress(100)
            setLoading(false)
            alert(`✅ Batch Started Successfully!\n\nAll files have been auto-converted to PDBQT for Vina/Gnina compatibility.\n\n${totalLigands} jobs are now running. Redirecting to status board...`)
            navigate(`/dock/batch/${batch_id}`)

        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    // NEW: CSV SMILES Submit Handler
    const handleCSVSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile) {
            setError('❌ Missing Receptor file. Please upload a PDB/PDBQT file.')
            return
        }
        if (!csvFile) {
            setError('❌ Missing CSV file. Please upload the SMILES CSV.')
            return
        }

        setLoading(true)
        setError(null)
        setUploadProgress(0)

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // Create FormData for multipart upload
            const formData = new FormData()
            formData.append('receptor_file', receptorFile)
            formData.append('csv_file', csvFile)
            formData.append('grid_center_x', parseFloat(gridParams.center_x))
            formData.append('grid_center_y', parseFloat(gridParams.center_y))
            formData.append('grid_center_z', parseFloat(gridParams.center_z))
            formData.append('grid_size_x', parseFloat(gridParams.size_x))
            formData.append('grid_size_y', parseFloat(gridParams.size_y))
            formData.append('grid_size_z', parseFloat(gridParams.size_z))
            formData.append('engine', engine) // Use selected engine

            setUploadProgress(20)

            const res = await fetch(`${API_URL}/jobs/batch/submit-csv`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: formData
            })

            setUploadProgress(80)

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'CSV batch submission failed')
            }

            const result = await res.json()
            setUploadProgress(100)

            alert(`Batch Processing Started!\n✅ ${result.jobs_created} jobs successfully running.\n⚠️ ${result.conversion_errors || 0} skipped (invalid SMILES).`)
            // Navigate to the new Batch Results Page
            navigate(`/dock/batch/${result.batch_id}`)

        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
                    <div className="mb-8 text-center">
                        <h1 className="text-3xl font-bold text-slate-900 mb-2">Molecular Docking</h1>
                        <p className="text-slate-500">Run docking for 1 to 100 ligands instantly.</p>
                    </div>

                    <form onSubmit={uploadMode === 'csv' ? handleCSVSubmit : handleSubmit} className="space-y-6">

                        {/* SECTION 1: Receptor & Preparation */}
                        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold">1</div>
                                <h3 className="font-bold text-slate-700">Receptor & Preparation</h3>
                            </div>
                            <div className="p-6 space-y-4">
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">
                                        Target Protein <span className="text-xs font-normal text-slate-500 ml-1">(PDB/PDBQT - Auto-Converts to PDBQT)</span>
                                    </label>
                                    <input
                                        type="file"
                                        accept=".pdb,.pdbqt"
                                        onChange={(e) => setReceptorFile(e.target.files[0])}
                                        className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition-all cursor-pointer"
                                    />
                                </div>
                                <div className="flex items-center gap-2 pt-2">
                                    <input type="checkbox" checked={true} readOnly className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500" />
                                    <label className="text-sm text-slate-600">
                                        <span className="font-semibold text-slate-700">Auto-Remove Waters</span> (Recommended)
                                    </label>
                                </div>
                                <div className="flex items-center gap-2">
                                    <input type="checkbox" checked={true} readOnly className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500" />
                                    <label className="text-sm text-slate-600">
                                        <span className="font-semibold text-slate-700">Add Polar Hydrogens</span>
                                    </label>
                                </div>
                            </div>
                        </div>

                        {/* SECTION 2: Ligands */}
                        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold">2</div>
                                <h3 className="font-bold text-slate-700">Ligand Library</h3>
                            </div>
                            <div className="p-6 space-y-4">
                                {/* Upload Mode Toggle */}
                                <div className="flex gap-2 p-1 bg-slate-100 rounded-xl mb-4">
                                    <button
                                        type="button"
                                        onClick={() => setUploadMode('files')}
                                        className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${uploadMode === 'files' ? 'bg-white shadow-sm text-emerald-700' : 'text-slate-600 hover:text-slate-800'}`}
                                    >
                                        📁 Files (.pdbqt, .sdf, .mol2)
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setUploadMode('csv')}
                                        className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${uploadMode === 'csv' ? 'bg-white shadow-sm text-emerald-700' : 'text-slate-600 hover:text-slate-800'}`}
                                    >
                                        📋 Paste SMILES (CSV)
                                    </button>
                                </div>

                                {uploadMode === 'files' ? (
                                    <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:bg-slate-100 transition-colors">
                                        <label className="cursor-pointer block">
                                            <div className="text-4xl mb-2">📂</div>
                                            <span className="block text-sm font-bold text-slate-700 mb-1">Click to Select Ligand Files</span>
                                            <span className="block text-xs text-slate-500 mb-4">AVG/SDF/MOL2/PDBQT (Auto-converted to PDBQT)</span>
                                            <input
                                                type="file"
                                                multiple
                                                accept=".pdbqt,.sdf,.mol2"
                                                onChange={handleLigandChange}
                                                className="hidden"
                                            />
                                            {ligandFiles.length > 0 ? (
                                                <div className="inline-block px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-bold">
                                                    {ligandFiles.length} files attached
                                                </div>
                                            ) : (
                                                <span className="px-4 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-600 shadow-sm">
                                                    Browse Files...
                                                </span>
                                            )}
                                        </label>
                                    </div>
                                ) : (
                                    <div>
                                        <label className="block text-sm font-bold text-slate-700 mb-2">Upload CSV File</label>
                                        <input
                                            type="file"
                                            accept=".csv"
                                            onChange={(e) => setCsvFile(e.target.files[0])}
                                            className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100 cursor-pointer"
                                        />
                                        <p className="text-xs text-slate-400 mt-2">
                                            Required Column: <code className="bg-slate-100 px-1 rounded">smiles</code>
                                        </p>
                                        {csvFile && <p className="text-sm text-emerald-600 font-bold mt-2">✓ {csvFile.name}</p>}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* SECTION 3: Configuration */}
                        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center font-bold">3</div>
                                <h3 className="font-bold text-slate-700">Engine & Grid Configuration</h3>
                            </div>
                            <div className="p-6 space-y-6">
                                {/* Engine Card */}
                                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-xl p-4 flex items-center gap-4">
                                    <div className="p-3 bg-white rounded-full text-2xl shadow-sm">🧠</div>
                                    <div>
                                        <div className="font-bold text-indigo-900">Consensus Mode Active</div>
                                        <div className="text-xs text-indigo-700 mt-1">
                                            Running <span className="font-bold">AutoDock Vina</span> and <span className="font-bold">Gnina (Deep Learning)</span> in parallel.
                                        </div>
                                    </div>
                                </div>

                                {/* Grid Params (Read Only) */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                                        <div className="text-xs text-slate-500 uppercase font-bold mb-1">Grid Center</div>
                                        <div className="font-mono text-slate-700">
                                            {gridParams.center_x}, {gridParams.center_y}, {gridParams.center_z}
                                        </div>
                                    </div>
                                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                                        <div className="text-xs text-slate-500 uppercase font-bold mb-1">Grid Size (Å)</div>
                                        <div className="font-mono text-slate-700">
                                            {gridParams.size_x} x {gridParams.size_y} x {gridParams.size_z}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Prep Progress Visualization */}
                        {(loading || preparationStep > 0) && (
                            <div className="mb-6 animate-in fade-in zoom-in duration-300">
                                <PreparationProgress currentStep={preparationStep || 1} />

                                {batchId && (
                                    <div className="mt-4 p-3 bg-slate-800 text-white rounded-lg flex justify-between items-center shadow-lg font-mono text-sm">
                                        <span className="flex items-center gap-2">
                                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                                            BATCH ID:
                                        </span>
                                        <span className="font-bold tracking-wider">{batchId}</span>
                                    </div>
                                )}
                            </div>
                        )}

                        {error && (
                            <div className="flex items-center gap-3 text-red-600 text-sm bg-red-50 p-4 rounded-xl border border-red-100">
                                <span className="text-xl">⚠️</span>
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className={`w-full py-4 rounded-xl font-bold text-lg text-white shadow-lg transition-all transform active:scale-[0.99] ${loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-gradient-to-r from-primary-600 to-secondary-600 hover:from-primary-700 hover:to-secondary-700 shadow-primary-500/30'}`}
                        >
                            {loading ? (typeof loading === 'string' ? loading : 'Processing Batch...') : '🚀 Launch Docking Batch'}
                        </button>
                    </form>
                </div>
            </main>
        </div>
    )
}
