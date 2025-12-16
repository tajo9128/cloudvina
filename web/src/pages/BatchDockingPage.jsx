import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import GridBoxConfigurator from '../components/GridBoxConfigurator'

export default function BatchDockingPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFiles, setLigandFiles] = useState([])
    const [error, setError] = useState(null)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [batchId, setBatchId] = useState(null)

    // NEW: Upload Mode Toggle (files vs csv)
    const [uploadMode, setUploadMode] = useState('files') // 'files' or 'csv'
    const [csvFile, setCsvFile] = useState(null)
    const [engine, setEngine] = useState('vina')

    // Grid Box State
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
                setError("Maximum 100 files allowed per batch.")
                return
            }
            setLigandFiles(files)
            setError(null)
        }
    }

    const uploadFile = async (url, file) => {
        const res = await fetch(url, {
            method: 'PUT',
            body: file,
            headers: {
                'Content-Type': file.type || 'application/octet-stream'
            }
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

            // 3. Start Batch
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
                    }
                })
            })

            if (!startRes.ok) throw new Error('Failed to start batch execution')

            setUploadProgress(100)
            alert(`Batch started successfully! ${totalLigands} jobs submitted.`)
            navigate('/dashboard')

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
        if (!receptorFile || !csvFile) {
            setError('Please select a receptor and a CSV file with SMILES.')
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
            formData.append('engine', engine)

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

            alert(`Batch submitted! ${result.jobs_created} jobs created. ${result.conversion_errors || 0} conversion errors.`)
            navigate('/dashboard')

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
                    <div className="mb-8 flex justify-between items-center">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 mb-2">Batch Docking</h1>
                            <p className="text-slate-500">Dock multiple ligands against a single target.</p>
                        </div>
                        <Link to="/dock/new" className="px-5 py-2.5 bg-slate-100 text-slate-700 font-bold rounded-xl hover:bg-slate-200 transition-colors flex items-center">
                            <span className="mr-2">🔬</span> Single Mode
                        </Link>
                    </div>

                    <form onSubmit={uploadMode === 'csv' ? handleCSVSubmit : handleSubmit} className="space-y-8">
                        {/* Upload Mode Toggle */}
                        <div className="flex gap-2 p-1 bg-slate-100 rounded-xl">
                            <button
                                type="button"
                                onClick={() => setUploadMode('files')}
                                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${uploadMode === 'files' ? 'bg-white shadow-sm text-primary-700' : 'text-slate-600 hover:text-slate-800'}`}
                            >
                                📁 Ligand Files
                            </button>
                            <button
                                type="button"
                                onClick={() => setUploadMode('csv')}
                                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${uploadMode === 'csv' ? 'bg-white shadow-sm text-primary-700' : 'text-slate-600 hover:text-slate-800'}`}
                            >
                                📋 CSV (SMILES)
                            </button>
                        </div>

                        {/* Receptor */}
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Receptor (PDB/PDBQT)</label>
                            <input
                                type="file"
                                accept=".pdb,.pdbqt"
                                onChange={(e) => setReceptorFile(e.target.files[0])}
                                className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                            />
                        </div>

                        {/* Conditional: Ligand Files or CSV */}
                        {uploadMode === 'files' ? (
                            <div>
                                <label className="block text-sm font-bold text-slate-700 mb-2">Ligands (Select Multiple, Max 100)</label>
                                <input
                                    type="file"
                                    multiple
                                    accept=".pdbqt,.sdf,.mol2"
                                    onChange={handleLigandChange}
                                    className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-secondary-50 file:text-secondary-700 hover:file:bg-secondary-100"
                                />
                                <p className="text-xs text-slate-400 mt-1">{ligandFiles.length} files selected</p>
                            </div>
                        ) : (
                            <div>
                                <label className="block text-sm font-bold text-slate-700 mb-2">CSV File with SMILES (Max 50 rows)</label>
                                <input
                                    type="file"
                                    accept=".csv"
                                    onChange={(e) => setCsvFile(e.target.files[0])}
                                    className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
                                />
                                <p className="text-xs text-slate-400 mt-1">
                                    CSV must have a <code className="bg-slate-100 px-1 rounded">smiles</code> column. Optional: <code className="bg-slate-100 px-1 rounded">name</code> column.
                                </p>
                                {csvFile && <p className="text-xs text-green-600 mt-1">✓ {csvFile.name} selected</p>}
                            </div>
                        )}

                        {/* Docking Engine (for CSV mode) */}
                        {uploadMode === 'csv' && (
                            <div>
                                <label className="block text-sm font-bold text-slate-700 mb-2">Docking Engine</label>
                                <select
                                    value={engine}
                                    onChange={(e) => setEngine(e.target.value)}
                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                                >
                                    <option value="vina">AutoDock Vina (Fast)</option>
                                    <option value="gnina">Gnina (AI-Powered)</option>
                                    <option value="consensus">Consensus (Both)</option>
                                </select>
                            </div>
                        )}

                        {/* Grid Box */}
                        <GridBoxConfigurator onConfigChange={setGridParams} initialConfig={gridParams} />

                        {/* Progress */}
                        {loading && (
                            <div className="w-full bg-slate-200 rounded-full h-2.5 mb-4">
                                <div className="bg-primary-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
                            </div>
                        )}

                        {error && <div className="text-red-600 text-sm bg-red-50 p-3 rounded-lg">{error}</div>}

                        <button
                            type="submit"
                            disabled={loading}
                            className={`w-full py-4 rounded-xl font-bold text-white shadow-lg ${loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700'}`}
                        >
                            {loading ? 'Processing Batch...' : uploadMode === 'csv' ? '🧪 Dock SMILES Compounds' : '🚀 Launch Batch Docking'}
                        </button>
                    </form>
                </div>
            </main>
        </div>
    )
}
