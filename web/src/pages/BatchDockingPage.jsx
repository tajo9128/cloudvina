import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
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
            if (files.length % 10 !== 0) {
                setError("Batch size must be a multiple of 10 (e.g., 10, 20, 30... 100).")
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

                    <form onSubmit={handleSubmit} className="space-y-8">
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

                        {/* Ligands */}
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
                            {loading ? 'Processing Batch...' : 'Launch Batch Docking'}
                        </button>
                    </form>
                </div>
            </main>
        </div>
    )
}
