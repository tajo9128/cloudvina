import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { API_URL } from '../config'

export default function ConverterPage() {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(false)

    // SEO: Set Title and Meta Tags
    useEffect(() => {
        document.title = "Free SDF to PDBQT Converter | BioDockify"

        // Helper to set meta tag
        const setMeta = (name, content) => {
            let element = document.querySelector(`meta[name="${name}"]`)
            if (!element) {
                element = document.createElement('meta')
                element.setAttribute('name', name)
                document.head.appendChild(element)
            }
            element.setAttribute('content', content)
        }

        setMeta('description', 'Convert SDF, MOL, PDB, and MOL2 files to PDBQT format instantly for AutoDock Vina. Free online tool for researchers and students.')
        setMeta('keywords', 'sdf to pdbqt, pdbqt converter, molecular docking, autodock vina, ligand preparation, zinc15, pubchem, rcsb pdb, BioDockify')
    }, [])

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

            const response = await fetch(`${API_URL}/tools/convert-to-pdbqt`, {
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
            a.download = file.name.replace(/\.(sdf|mol|pdb|mol2)$/i, '') + '.pdbqt'
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
        <div className="bg-slate-50 min-h-screen pt-32 pb-20">
            {/* Hero Section */}
            <section className="text-center relative z-10 mb-16">
                <div className="container mx-auto px-4">
                    <div className="inline-block bg-primary-50 border border-primary-100 text-primary-600 px-4 py-1 rounded-full text-sm font-bold mb-6">
                        🧪 Free Research Tool
                    </div>
                    <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight tracking-tight text-slate-900">
                        Convert Molecules to <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-secondary-500">PDBQT Format Instantly</span>
                    </h1>
                    <p className="text-xl text-slate-600 mb-8 max-w-2xl mx-auto">
                        Prepare your ligands for AutoDock Vina. Supports SDF, PDB, MOL, and MOL2 formats.
                        Adds polar hydrogens and partial charges automatically.
                    </p>
                </div>
            </section>

            {/* Converter Tool */}
            <section className="pb-20 relative z-10">
                <div className="container mx-auto px-4">
                    <div className="max-w-2xl mx-auto bg-white rounded-2xl p-8 md:p-12 shadow-xl shadow-primary-900/5 border border-slate-200">
                        <form onSubmit={handleConvert} className="space-y-8">
                            <div className="border-2 border-dashed border-slate-300 rounded-xl p-10 text-center hover:border-primary-500 hover:bg-primary-50/50 transition-all cursor-pointer relative group bg-slate-50">
                                <input
                                    type="file"
                                    accept=".sdf,.mol,.pdb,.mol2"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                                />
                                <div className="space-y-4 relative z-10">
                                    <div className="text-6xl mb-4 transform group-hover:scale-110 transition-transform duration-300">📄</div>
                                    {file ? (
                                        <div>
                                            <div className="text-xl font-bold text-slate-900 mb-1">{file.name}</div>
                                            <div className="text-sm text-primary-600 font-bold">Ready to convert</div>
                                        </div>
                                    ) : (
                                        <>
                                            <div className="text-xl font-bold text-slate-900">Click to Upload Molecule</div>
                                            <div className="text-sm text-slate-500">Supports .sdf, .pdb, .mol, .mol2</div>
                                        </>
                                    )}
                                </div>
                            </div>

                            {error && (
                                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl text-sm flex items-center">
                                    <span className="mr-2 text-xl">⚠️</span> {error}
                                </div>
                            )}

                            {success && (
                                <div className="bg-green-50 border border-green-200 text-green-700 p-4 rounded-xl text-sm flex items-center justify-center">
                                    <span className="mr-2 text-xl">✅</span> Conversion successful! Download started.
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={!file || loading}
                                className={`w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-lg ${!file || loading
                                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                                    : 'btn-primary hover:shadow-primary-600/30'
                                    }`}
                            >
                                {loading ? (
                                    <span className="flex items-center justify-center">
                                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Processing...
                                    </span>
                                ) : (
                                    'Convert to PDBQT Now'
                                )}
                            </button>
                        </form>
                        <div className="mt-6 text-center text-xs text-slate-400">
                            Powered by RDKit & Meeko • Secure Cloud Processing
                        </div>
                    </div>
                </div>
            </section>

            {/* Content Section */}
            <section className="py-20 bg-white border-t border-slate-200">
                <div className="container mx-auto px-4 max-w-4xl">

                    {/* Importance */}
                    <div className="mb-16">
                        <h2 className="text-3xl font-bold text-slate-900 mb-6 border-l-4 border-primary-500 pl-4">Why is PDBQT Format Important?</h2>
                        <div className="prose prose-lg prose-slate text-slate-600">
                            <p className="mb-4">
                                <strong className="text-primary-600">AutoDock Vina</strong>, the industry-standard software for molecular docking, requires both the receptor (protein) and ligand (drug candidate) to be in <strong>PDBQT</strong> format.
                            </p>
                            <p>
                                PDBQT stands for <em>Protein Data Bank, Partial Charge (Q), and Atom Type (T)</em>. Unlike standard SDF or PDB files, a PDBQT file contains crucial physicochemical information needed for docking simulations:
                            </p>
                            <ul className="list-disc pl-6 space-y-2 mt-4 mb-6 text-slate-700">
                                <li><strong className="text-slate-900">Polar Hydrogens:</strong> Essential for hydrogen bonding interactions.</li>
                                <li><strong className="text-slate-900">Partial Charges:</strong> Calculated using the Gasteiger method to simulate electrostatic forces.</li>
                                <li><strong className="text-slate-900">Rotatable Bonds:</strong> Defines which parts of the molecule can flex and move during docking.</li>
                            </ul>
                            <p>
                                Most chemical databases provide files in 2D or 3D SDF format. BioDockify's converter bridges this gap, automatically preparing your files for high-accuracy docking.
                            </p>
                        </div>
                    </div>

                    {/* Download Guide */}
                    <div className="mb-16">
                        <h2 className="text-3xl font-bold text-slate-900 mb-6 border-l-4 border-secondary-500 pl-4">Where to Download Ligand Molecules?</h2>
                        <p className="text-slate-600 mb-8 text-lg">
                            Here are the top 3 free databases to find 3D structures for your research:
                        </p>

                        <div className="grid md:grid-cols-3 gap-6">
                            {/* ZINC15 */}
                            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 hover:border-primary-300 transition-colors">
                                <h3 className="text-xl font-bold text-slate-900 mb-2">1. ZINC15 Database</h3>
                                <p className="text-sm text-slate-500 mb-4">Best for virtual screening and commercially available compounds.</p>
                                <ul className="text-sm text-slate-600 space-y-2 mb-4">
                                    <li>• Go to <a href="https://zinc15.docking.org" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">zinc15.docking.org</a></li>
                                    <li>• Search for compound</li>
                                    <li>• Download <strong>"3D SDF"</strong></li>
                                </ul>
                            </div>

                            {/* PubChem */}
                            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 hover:border-primary-300 transition-colors">
                                <h3 className="text-xl font-bold text-slate-900 mb-2">2. PubChem</h3>
                                <p className="text-sm text-slate-500 mb-4">World's largest free collection of chemical information.</p>
                                <ul className="text-sm text-slate-600 space-y-2 mb-4">
                                    <li>• Go to <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">PubChem</a></li>
                                    <li>• Click "Download"</li>
                                    <li>• Select <strong>"3D Conformer" (SDF)</strong></li>
                                </ul>
                            </div>

                            {/* RCSB PDB */}
                            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 hover:border-primary-300 transition-colors">
                                <h3 className="text-xl font-bold text-slate-900 mb-2">3. RCSB PDB</h3>
                                <p className="text-sm text-slate-500 mb-4">For extracting co-crystallized ligands from protein complexes.</p>
                                <ul className="text-sm text-slate-600 space-y-2 mb-4">
                                    <li>• Go to <a href="https://www.rcsb.org" target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">rcsb.org</a></li>
                                    <li>• Find PDB Entry</li>
                                    <li>• Download <strong>"Ligand"</strong> file</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Step by Step */}
                    <div className="bg-slate-900 p-8 rounded-2xl text-white">
                        <h2 className="text-2xl font-bold text-white mb-6">How to Use This Tool</h2>
                        <div className="grid md:grid-cols-3 gap-8 text-center">
                            <div>
                                <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center text-white font-bold text-xl mx-auto mb-4 shadow-lg shadow-primary-500/30">1</div>
                                <h3 className="font-bold text-white mb-2">Upload File</h3>
                                <p className="text-sm text-slate-400">Select your .sdf, .mol, or .pdb file from your computer.</p>
                            </div>
                            <div>
                                <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center text-white font-bold text-xl mx-auto mb-4 shadow-lg shadow-primary-500/30">2</div>
                                <h3 className="font-bold text-white mb-2">Convert</h3>
                                <p className="text-sm text-slate-400">Our cloud servers process the file using Meeko & RDKit.</p>
                            </div>
                            <div>
                                <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center text-white font-bold text-xl mx-auto mb-4 shadow-lg shadow-primary-500/30">3</div>
                                <h3 className="font-bold text-white mb-2">Download</h3>
                                <p className="text-sm text-slate-400">Get your .pdbqt file instantly, ready for docking.</p>
                            </div>
                        </div>
                    </div>

                    {/* CTA */}
                    <div className="mt-16 text-center">
                        <h2 className="text-3xl font-bold text-slate-900 mb-6">Ready to Run Your Docking Simulation?</h2>
                        <p className="text-slate-600 mb-8 text-lg">
                            Now that you have your PDBQT files, start your molecular docking job on BioDockify.
                        </p>
                        <Link to="/dock/new" className="btn-primary text-lg px-10 py-4 rounded-xl inline-block shadow-lg shadow-primary-600/20">
                            Start Docking Job →
                        </Link>
                    </div>

                </div>
            </section>
        </div>
    )
}
