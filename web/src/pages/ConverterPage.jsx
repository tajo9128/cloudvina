import { useState } from 'react'
import { API_URL } from '../config'

export default function ConverterPage() {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(false)

    const handleFileChange = (e) => {
        if (e.target.files[0]) {
            setFile(e.target.files[0])
            setError(null)
            setSuccess(false)
        }
    }

    const handleConvert = async (e) => {
        e.preventDefault()
        if (!file) return

        setLoading(true)
        setError(null)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch(`${API_URL}/tools/convert-sdf`, {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Conversion failed')
            }

            // Handle file download
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = file.name.replace('.sdf', '.pdbqt')
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)

            setSuccess(true)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <div className="container mx-auto px-4 py-12">
                <div className="max-w-2xl mx-auto">
                    <div className="text-center mb-10">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">SDF to PDBQT Converter</h2>
                        <p className="text-gray-600">
                            Convert your ligand files from SDF format to PDBQT format instantly.
                            Ready for AutoDock Vina docking.
                        </p>
                    </div>

                    <div className="bg-white rounded-xl shadow-lg p-8">
                        <form onSubmit={handleConvert} className="space-y-6">
                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-500 transition-colors cursor-pointer relative">
                                <input
                                    type="file"
                                    accept=".sdf"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />
                                <div className="space-y-2">
                                    <div className="text-4xl mb-2">📄</div>
                                    {file ? (
                                        <div className="text-purple-600 font-semibold">{file.name}</div>
                                    ) : (
                                        <>
                                            <div className="text-gray-600 font-medium">Click to upload SDF file</div>
                                            <div className="text-sm text-gray-400">or drag and drop</div>
                                        </>
                                    )}
                                </div>
                            </div>

                            {error && (
                                <div className="bg-red-50 text-red-600 p-4 rounded-lg text-sm">
                                    {error}
                                </div>
                            )}

                            {success && (
                                <div className="bg-green-50 text-green-600 p-4 rounded-lg text-sm flex items-center justify-center">
                                    <span className="mr-2">✓</span> Conversion successful! Download started.
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={!file || loading}
                                className={`w-full py-3 rounded-lg font-semibold text-white transition ${!file || loading
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-purple-600 hover:bg-purple-700 shadow-md'
                                    }`}
                            >
                                {loading ? (
                                    <span className="flex items-center justify-center">
                                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Converting...
                                    </span>
                                ) : (
                                    'Convert to PDBQT'
                                )}
                            </button>
                        </form>
                    </div>

                    <div className="mt-8 text-center text-sm text-gray-500">
                        <p>Powered by RDKit and Meeko. Runs securely on our cloud servers.</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
