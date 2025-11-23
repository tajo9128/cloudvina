import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'

export default function NewJobPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFile, setLigandFile] = useState(null)
    const [error, setError] = useState(null)

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!receptorFile || !ligandFile) {
            setError('Please select both receptor and ligand files')
            return
        }

        setLoading(true)
        setError(null)

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // 1. Create Job
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const createRes = await fetch(`${apiUrl}/jobs/submit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    receptor_filename: receptorFile.name,
                    ligand_filename: ligandFile.name
                })
            })

            if (!createRes.ok) {
                const err = await createRes.json()
                throw new Error(err.detail || 'Failed to create job')
            }

            const { job_id, upload_urls } = await createRes.json()

            // 2. Upload Files
            await uploadFile(upload_urls.receptor, receptorFile)
            await uploadFile(upload_urls.ligand, ligandFile)

            // 3. Start Job
            const startRes = await fetch(`${apiUrl}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!startRes.ok) throw new Error('Failed to start job')

            navigate(`/dock/${job_id}`)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
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

    return (
        <div className="min-h-screen bg-gray-50">


            <main className="container mx-auto px-4 py-12">
                <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
                    <h1 className="text-2xl font-bold text-gray-900 mb-6">Start New Docking Job</h1>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Receptor Upload */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Receptor (PDB)
                            </label>
                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-purple-500 transition-colors cursor-pointer relative">
                                <input
                                    type="file"
                                    accept=".pdb"
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />
                                <div className="space-y-1">
                                    <div className="text-3xl">🧬</div>
                                    {receptorFile ? (
                                        <div className="text-purple-600 font-medium">{receptorFile.name}</div>
                                    ) : (
                                        <div className="text-gray-500">Upload Receptor (.pdb)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Ligand Upload */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Ligand (PDBQT, SDF, MOL2)
                            </label>
                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-purple-500 transition-colors cursor-pointer relative">
                                <input
                                    type="file"
                                    accept=".pdbqt,.sdf,.mol2"
                                    onChange={(e) => setLigandFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />
                                <div className="space-y-1">
                                    <div className="text-3xl">💊</div>
                                    {ligandFile ? (
                                        <div className="text-purple-600 font-medium">{ligandFile.name}</div>
                                    ) : (
                                        <div className="text-gray-500">Upload Ligand (.pdbqt, .sdf)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {error && (
                            <div className="bg-red-50 text-red-600 p-4 rounded-lg text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className={`w-full py-3 rounded-lg font-bold text-white transition ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700 shadow-md'
                                }`}
                        >
                            {loading ? 'Submitting...' : 'Launch Docking Job'}
                        </button>
                    </form>
                </div>
            </main>
        </div>
    )
}
