import { supabase } from '../supabaseClient'

export default function ExportButtons({ jobId = null, className = "" }) {
    const handleExport = async (format) => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) {
                alert('Please log in to export data')
                return
            }

            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const endpoint = jobId
                ? `${apiUrl}/jobs/${jobId}/export/${format}`
                : `${apiUrl}/jobs/export/${format}`

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
            let filename = `cloudvina_export.${format}`
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
                onClick={() => handleExport('csv')}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition font-medium text-sm"
                title="Export to CSV"
            >
                📊 CSV
            </button>
            <button
                onClick={() => handleExport('json')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium text-sm"
                title="Export to JSON"
            >
                📄 JSON
            </button>
            {jobId && (
                <button
                    onClick={() => handleExport('pdf')}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition font-medium text-sm"
                    title="Export to PDF"
                >
                    📑 PDF
                </button>
            )}
        </div>
    )
}
