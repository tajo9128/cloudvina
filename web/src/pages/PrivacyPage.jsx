export default function PrivacyPage() {
    return (
        <div className="min-h-screen bg-slate-50 pt-32 pb-20">
            <div className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto bg-white rounded-2xl p-8 md:p-12 shadow-sm border border-slate-200">
                    <h1 className="text-4xl font-bold text-slate-900 mb-8 tracking-tight">Privacy Policy</h1>
                    <div className="prose prose-lg max-w-none text-slate-600">
                        <p className="text-slate-500 mb-8 text-sm">Last updated: November 23, 2025</p>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">1. Information We Collect</h3>
                        <p className="mb-6 leading-relaxed">
                            We collect information you provide directly to us when using our **Zero-Cost Drug Discovery Pipeline**. This includes:
                        </p>
                        <ul className="list-disc pl-6 mb-6 space-y-2">
                            <li><strong>Account Information:</strong> Name, email address, and institutional affiliation (optional).</li>
                            <li><strong>Research Data:</strong> Molecular structures (PDB, PDBQT, SDF, MOL2), protein targets, and ligand libraries uploaded for processing across the 7 phases.</li>
                            <li><strong>Simulation Parameters:</strong> Configuration settings for Docking (Grid Box), MD Simulations (Temperature, Steps), and Analysis.</li>
                        </ul>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">2. How We Use Your Information</h3>
                        <p className="mb-6 leading-relaxed">
                            We use your information strictly to operate the 7-Phase Pipeline:
                        </p>
                        <ul className="list-disc pl-6 mb-6 space-y-2">
                            <li><strong>Phases 1-3 (Docking & Simulation):</strong> Processing molecular inputs to generate binding poses and trajectories.</li>
                            <li><strong>Phases 4-6 (Analysis & Ranking):</strong> calculating binding free energies (MM-GBSA), performing consensus ranking, and predicting ADMET properties.</li>
                            <li><strong>Phase 7 (Reporting):</strong> Generating downloadable PDF reports of your lead candidates.</li>
                        </ul>
                        <p className="mb-6 leading-relaxed">
                            We do <strong>not</strong> use your proprietary research data for model training or share it with third parties.
                        </p>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">3. Data Security & Retention</h3>
                        <p className="mb-6 leading-relaxed">
                            Your research integrity is paramount.
                        </p>
                        <ul className="list-disc pl-6 mb-6 space-y-2">
                            <li><strong>Ephemeral Processing:</strong> All heavy computation (Docking, MD) occurs on ephemeral worker nodes (e.g., Google Colab). Data is deleted from these nodes immediately after the job completes.</li>
                            <li><strong>Storage:</strong> Input files and generated results are stored securely in your private cloud bucket and are accessible only by you.</li>
                            <li><strong>Encryption:</strong> All data transfer is encrypted via SSL/TLS.</li>
                        </ul>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">4. Contact Us</h3>
                        <p className="mb-6 leading-relaxed">
                            If you have any questions about this Privacy Policy, please contact us at <a href="mailto:biodockify@hotmail.com" className="text-primary-600 hover:text-primary-700 underline decoration-primary-200 hover:decoration-primary-500 transition-all">biodockify@hotmail.com</a>.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
