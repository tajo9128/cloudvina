import { supabase } from '../supabaseClient'
import { API_URL } from '../config'

export default function ExportButtons({ jobId = null, job = null, className = "" }) {
    const handleExport = async (format) => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return alert('Please sign in to export data')

            const endpoint = jobId ? `${API_URL}/jobs/${jobId}/export/${format}` : `${API_URL}/export/${format}`
            const response = await fetch(endpoint, { headers: { 'Authorization': `Bearer ${session.access_token}` } })
            if (!response.ok) throw new Error('Export failed')

            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            // Filename logic
            let filename = `BioDockify_export.${format}`
            const disp = response.headers.get('Content-Disposition')
            if (disp && disp.includes('filename=')) filename = disp.split('filename=')[1].replace(/"/g, '')

            a.download = filename
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (error) {
            console.error('Export error:', error)
            alert('Export failed.')
        }
    }

    // Direct Download Helper
    const downloadUrl = (url) => { if (url) window.location.href = url }

    return (
        <div className={`flex items-center gap-2 ${className}`}>
            {/* 1. PDF Report */}


            {/* 2. Output PDBQT */}
            {job?.download_urls?.output_vina && (
                <button onClick={() => downloadUrl(job.download_urls.output_vina)} className="px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 transition-all font-bold text-xs flex items-center gap-2 border border-indigo-200">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" /></svg>
                    PDBQT
                </button>
            )}

            {/* 3. Consensus JSON */}
            {job?.download_urls?.results_json && (
                <button onClick={() => downloadUrl(job.download_urls.results_json)} className="px-4 py-2 bg-emerald-50 text-emerald-700 rounded-lg hover:bg-emerald-100 transition-all font-bold text-xs flex items-center gap-2 border border-emerald-200">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
                    JSON
                </button>
            )}

            {/* 4. PyMOL Script */}
            <button onClick={() => handleExport('pymol')} className="px-4 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-all font-bold text-xs flex items-center gap-2 border border-blue-200">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                PyMOL
            </button>

            {/* 5. Config & Log Dropdown (Simplified as separate buttons for now) */}
            {job?.download_urls?.config && (
                <button onClick={() => downloadUrl(job.download_urls.config)} className="px-3 py-2 text-slate-500 hover:text-slate-800 transition-all" title="Download Config">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                </button>
            )}
            {job?.download_urls?.log && (
                <button onClick={() => downloadUrl(job.download_urls.log)} className="px-3 py-2 text-slate-500 hover:text-slate-800 transition-all" title="Download Log">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                </button>
            )}
        </div>
    )
}
