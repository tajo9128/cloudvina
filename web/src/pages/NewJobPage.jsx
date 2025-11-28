import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'

import { API_URL } from '../config'

export default function NewJobPage() {
    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)
    const [receptorFile, setReceptorFile] = useState(null)
    const [ligandFile, setLigandFile] = useState(null)
    const [error, setError] = useState(null)



    // Job Progress State
    const [submittedJob, setSubmittedJob] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const ESTIMATED_DURATION = 300 // 5 minutes





    useEffect(() => {
        let timer
        if (submittedJob && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status)) {
            const startTime = submittedJob.created_at ? new Date(submittedJob.created_at).getTime() : Date.now()
            timer = setInterval(() => {
                const now = Date.now()
                setElapsedTime(Math.floor((now - startTime) / 1000))
            }, 1000)
        }
        return () => clearInterval(timer)
    }, [submittedJob])

    // Poll for job status
    useEffect(() => {
        let poller
        if (submittedJob && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status)) {
            poller = setInterval(async () => {
                try {
                    const { data: { session } } = await supabase.auth.getSession()
                    if (!session) return

                    const res = await fetch(`${API_URL}/jobs/${submittedJob.job_id}`, {
                        headers: {
                            'Authorization': `Bearer ${session.access_token}`
                        }
                    })

                    if (res.ok) {
                        const updatedJob = await res.json()
                        setSubmittedJob(prev => ({
                            ...prev,
                            status: updatedJob.status
                        }))
                    }
                } catch (err) {
                    console.error('Polling error:', err)
                }
            }, 5000)
        }
        return () => clearInterval(poller)
    }, [submittedJob])

    const getProgressColor = (percentage) => {
        if (percentage < 30) return 'bg-blue-500'
        if (percentage < 70) return 'bg-purple-500'
        return 'bg-green-500'
    }

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
    }

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
            const createRes = await fetch(`${API_URL}/jobs/submit`, {
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
                const errorMessage = typeof err.detail === 'object'
                    ? JSON.stringify(err.detail)
                    : (err.detail?.message || err.detail || 'Failed to create job')
                throw new Error(errorMessage)
            }

            const { job_id, upload_urls } = await createRes.json()

            // 2. Upload Files
            await uploadFile(upload_urls.receptor, receptorFile)
            await uploadFile(upload_urls.ligand, ligandFile)

            // 3. Start Job
            const startRes = await fetch(`${API_URL}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!startRes.ok) throw new Error('Failed to start job')

            // Instead of navigating, set state to show progress
            setSubmittedJob({
                job_id,
                status: 'SUBMITTED',
                created_at: new Date().toISOString()
            })

        } catch (err) {
            // Check if it's an email verification error
            if (err.message.includes('verify your email')) {
                setError(
                    <div className="flex flex-col items-start gap-2">
                        <span>{err.message}</span>
                        <button
                            onClick={async () => {
                                setLoading(true)
                                const { data, error } = await supabase.auth.refreshSession()
                                if (!error && data.session?.user?.email_confirmed_at) {
                                    setError(null)
                                    alert('Email verified successfully! You can now submit jobs.')
                                } else {
                                    alert('Email still not verified. Please check your inbox or try logging out and back in.')
                                }
                                setLoading(false)
                            }}
                            className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded border border-red-200 hover:bg-red-200 transition"
                        >
                            🔄 I have verified my email
                        </button>
                    </div>
                )
            } else {
                setError(err.message)
            }
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

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-gradient-to-b from-deep-navy-900 to-royal-blue-800">


            <main className="container mx-auto px-4 py-12">
                <div className="max-w-2xl mx-auto bg-white rounded-2xl shadow-xl border border-gray-100-light p-8">
                    <h1 className="text-2xl font-bold text-deep-navy-900 mb-6">Start New Docking Job</h1>



                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Receptor Upload */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Receptor (PDB)
                            </label>
                            <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors relative ${submittedJob ? 'bg-gray-50 border-gray-200' : 'border-gray-300 hover:border-purple-500 cursor-pointer'}`}>
                                <input
                                    type="file"
                                    accept=".pdb"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
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
                            <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors relative ${submittedJob ? 'bg-gray-50 border-gray-200' : 'border-gray-300 hover:border-purple-500 cursor-pointer'}`}>
                                <input
                                    type="file"
                                    accept=".pdbqt,.sdf,.mol2"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setLigandFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
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

                        {!submittedJob ? (
                            <button
                                type="submit"
                                disabled={loading}
                                className={`btn-blue-glow w-full ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                {loading ? 'Submitting...' : 'Launch Docking Job'}
                            </button>
                        ) : (
                            <div className="space-y-4 animate-in fade-in duration-500">
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="font-medium text-gray-700">Status: {submittedJob.status}</span>
                                        <span className="text-sm text-gray-500">Est. Remaining: {formatTime(remainingSeconds)}</span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="w-full bg-gray-200 rounded-full h-3 mb-2 overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-500 ${getProgressColor(progressPercentage)}`}
                                            style={{ width: `${progressPercentage}%` }}
                                        >
                                            {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status) && (
                                                <div className="w-full h-full animate-pulse bg-white/20"></div>
                                            )}
                                        </div>
                                    </div>

                                    <p className="text-xs text-gray-500 text-center">
                                        Job ID: {submittedJob.job_id}
                                    </p>
                                </div>

                                {submittedJob.status === 'SUCCEEDED' && (
                                    <button
                                        type="button"
                                        onClick={() => navigate(`/dock/${submittedJob.job_id}`)}
                                        className="w-full py-3 rounded-lg font-bold text-white bg-green-600 hover:bg-green-700 shadow-md transition animate-bounce"
                                    >
                                        View Results 🎉
                                    </button>
                                )}

                                {submittedJob.status === 'FAILED' && (
                                    <div className="text-center text-red-600 font-medium">
                                        Job Failed. Please try again.
                                    </div>
                                )}
                            </div>
                        )}
                    </form>
                </div>
            </main>
        </div>
    )
}
