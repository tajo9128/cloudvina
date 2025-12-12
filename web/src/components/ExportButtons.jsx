import { supabase } from '../supabaseClient'
import { API_URL } from '../config'

export default function ExportButtons({ jobId = null, className = "" }) {
    const handleExport = async (format) => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) {
                alert('Please sign in to export data')
                return
            }

            const endpoint = jobId
                ? `${API_URL}/jobs/${jobId}/export/${format}`
                : `${API_URL}/export/${format}`

            const response = await fetch(endpoint, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!response.ok) {
                throw new Error('Export failed')
            }

            // Get filename from Content-Disposition header or create default
            const contentDisposition = response.headers.get('Content-Disposition')
            let filename = `BioDockify_export.${format}`
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?(.+)"?/)
                if (filenameMatch) {
                    filename = filenameMatch[1]
                }
            }

            // Download file
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = filename
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)

        } catch (error) {
            console.error('Export error:', error)
            alert('Failed to export data. Please try again.')
        }
    }

    return (
        <div className={`flex gap-2 ${className}`}>
            <button
                onClick={() => handleExport('pdf')}
                className="px-6 py-3 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg hover:from-red-700 hover:to-red-800 transition-all shadow-lg hover:shadow-xl font-semibold text-sm flex items-center gap-2"
                title="Download Publication-Ready PDF Report"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path>
                </svg>
                📑 Download PDF Report
            </button>
        </div>
    )
}
