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

    // Helper: Fetch with retry logic for cold starts
    const fetchWithRetry = async (url, options, retries = 3, backoff = 1000) => {
        try {
            const res = await fetch(url, options)
            if (!res.ok) {
                // If 503 or 504 (Service Unavailable/Gateway Timeout), retry
                if ([503, 504].includes(res.status) && retries > 0) {
                    console.log(`Retrying ${url}... Attempts left: ${retries}`)
                    await new Promise(r => setTimeout(r, backoff))
                    return fetchWithRetry(url, options, retries - 1, backoff * 2)
                }
                return res // Return error response to be handled by caller
            }
            return res
        } catch (err) {
            // Retry on network errors (like Failed to fetch)
            if (retries > 0) {
                console.log(`Network error, retrying ${url}... Attempts left: ${retries}`)
                await new Promise(r => setTimeout(r, backoff))
                return fetchWithRetry(url, options, retries - 1, backoff * 2)
            }
            throw err
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!receptorFile || !ligandFile) {
            setError('Please select both receptor and ligand files')
            return
        }

        setLoading(true)
        setError('⏰ Connecting to server... (This may take up to 60s if server is sleeping)')

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            // 1. Create Job with Retry
            console.log('Step 1: Creating job with backend...')
            const createRes = await fetchWithRetry(`${API_URL}/jobs/submit`, {
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
            console.log('Step 1 response:', createRes.status, createRes.ok)

            if (!createRes.ok) {
                const err = await createRes.json()
                const errorMessage = typeof err.detail === 'object'
                    ? JSON.stringify(err.detail)
                    : (err.detail?.message || err.detail || 'Failed to create job')
                throw new Error(errorMessage)
            }

            const { job_id, upload_urls } = await createRes.json()
            console.log('Step 1 complete. Job ID:', job_id)

            // 2. Upload Files (No retry needed usually, S3 is reliable)
            console.log('Step 2: Uploading files to S3...')
            setError('📤 Uploading files...')
            await uploadFile(upload_urls.receptor, receptorFile)
            console.log('Receptor uploaded')
            await uploadFile(upload_urls.ligand, ligandFile)
            console.log('Ligand uploaded')

            // 3. Start Job with Retry
            console.log('Step 3: Starting job...')
            setError('🚀 Starting simulation...')
            const startRes = await fetchWithRetry(`${API_URL}/jobs/${job_id}/start`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })
            console.log('Step 3 response:', startRes.status, startRes.ok)

            if (!startRes.ok) throw new Error('Failed to start job')

            console.log('Job submitted successfully!')
            // Instead of navigating, set state to show progress
            setSubmittedJob({
                job_id,
                status: 'SUBMITTED',
                created_at: new Date().toISOString()
            })

        } catch (err) {
            console.error('Submission error:', err)
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
                setError(`Error: ${err.message}. Please try again.`)
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
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-2xl mx-auto glass-modern rounded-2xl p-8">
                    <h1 className="text-3xl font-extrabold text-white mb-6 tracking-tight">Start New Docking Job</h1>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Receptor Upload */}
                        <div>
                            <label className="block text-sm font-bold text-white mb-2">
                                Receptor (PDB)
                            </label>
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all relative group ${submittedJob ? 'bg-blue-900/20 border-blue-800' : 'border-blue-700/50 hover:border-cyan-400/50 hover:bg-blue-900/30 cursor-pointer bg-blue-900/10'}`}>
                                <input
                                    type="file"
                                    accept=".pdb"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setReceptorFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-20"
                                />
                                <div className="space-y-2 relative z-10">
                                    <div className="text-4xl mb-2 transform group-hover:scale-110 transition-transform duration-300 drop-shadow-[0_0_10px_rgba(0,217,255,0.3)]">🧬</div>
                                    {receptorFile ? (
                                        <div className="text-cyan-400 font-bold text-lg">{receptorFile.name}</div>
                                    ) : (
                                        <div className="text-blue-200 font-medium">Upload Receptor (.pdb)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Ligand Upload */}
                        <div>
                            <label className="block text-sm font-bold text-white mb-2">
                                Ligand (PDBQT, SDF, MOL2)
                            </label>
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all relative group ${submittedJob ? 'bg-blue-900/20 border-blue-800' : 'border-blue-700/50 hover:border-cyan-400/50 hover:bg-blue-900/30 cursor-pointer bg-blue-900/10'}`}>
                                <input
                                    type="file"
                                    accept=".pdbqt,.sdf,.mol2"
                                    disabled={!!submittedJob}
                                    onChange={(e) => setLigandFile(e.target.files[0])}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-20"
                                />
                                <div className="space-y-2 relative z-10">
                                    <div className="text-4xl mb-2 transform group-hover:scale-110 transition-transform duration-300 drop-shadow-[0_0_10px_rgba(0,217,255,0.3)]">💊</div>
                                    {ligandFile ? (
                                        <div className="text-cyan-400 font-bold text-lg">{ligandFile.name}</div>
                                    ) : (
                                        <div className="text-blue-200 font-medium">Upload Ligand (.pdbqt, .sdf)</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {error && (
                            <div className="bg-red-500/20 border border-red-500/50 text-red-200 p-4 rounded-xl text-sm backdrop-blur-sm">
                                {error}
                            </div>
                        )}

                        {!submittedJob ? (
                            <button
                                type="submit"
                                disabled={loading}
                                className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg ${loading
                                    ? 'bg-blue-900/50 text-blue-400/50 cursor-not-allowed border border-blue-800'
                                    : 'btn-cyan hover:shadow-cyan-500/40'}`}
                            >
                                {loading ? (
                                    <span className="flex items-center justify-center">
                                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Submitting...
                                    </span>
                                ) : 'Launch Docking Job'}
                            </button>
                        ) : (
                            <div className="space-y-4 animate-in fade-in duration-500">
                                <div className="bg-blue-900/30 rounded-xl p-6 border border-blue-700/50 backdrop-blur-sm">
                                    <div className="flex justify-between items-center mb-3">
                                        <span className="font-bold text-white">Status: <span className="text-cyan-400">{submittedJob.status}</span></span>
                                        <span className="text-sm text-blue-200">Est. Remaining: {formatTime(remainingSeconds)}</span>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="w-full bg-blue-900/50 rounded-full h-3 mb-3 overflow-hidden border border-blue-700/30">
                                        <div
                                            className={`h-full transition-all duration-500 bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_10px_rgba(0,217,255,0.5)]`}
                                            style={{ width: `${progressPercentage}%` }}
                                        >
                                            {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(submittedJob.status) && (
                                                <div className="w-full h-full animate-pulse bg-white/30"></div>
                                            )}
                                        </div>
                                    </div>

                                    <p className="text-xs text-blue-300/60 text-center font-mono">
                                        Job ID: {submittedJob.job_id}
                                    </p>
                                </div>

                                {submittedJob.status === 'SUCCEEDED' && (
                                    <button
                                        type="button"
                                        onClick={() => navigate(`/dock/${submittedJob.job_id}`)}
                                        className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 shadow-lg shadow-green-500/30 transition animate-bounce"
                                    >
                                        View Results 🎉
                                    </button>
                                )}

                                {submittedJob.status === 'FAILED' && (
                                    <div className="text-center text-red-400 font-bold bg-red-500/10 p-4 rounded-xl border border-red-500/30">
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
