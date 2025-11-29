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
                            We collect information you provide directly to us, such as when you create an account, submit a job, or contact us. This may include your name, email address, phone number, and institutional affiliation.
                        </p>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">2. How We Use Your Information</h3>
                        <p className="mb-6 leading-relaxed">
                            We use your information to provide, maintain, and improve our services, process your docking jobs, and communicate with you.
                        </p>

                        <h3 className="text-2xl font-bold text-slate-900 mt-10 mb-4">3. Data Security</h3>
                        <p className="mb-6 leading-relaxed">
                            We implement appropriate technical and organizational measures to protect your data. Your molecular data (receptors and ligands) is processed in secure, isolated containers and is deleted from our processing servers after job completion.
                        </p>

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
