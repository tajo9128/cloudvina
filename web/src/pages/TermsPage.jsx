export default function TermsPage() {
    return (
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <div className="container mx-auto px-4">
                <div className="max-w-3xl mx-auto glass-modern rounded-2xl p-8 md:p-12">
                    <h1 className="text-4xl font-extrabold text-white mb-8 tracking-tight">Terms of Service</h1>
                    <div className="prose prose-invert prose-lg max-w-none">
                        <p className="text-blue-200 mb-8">Last updated: November 23, 2025</p>

                        <h3 className="text-2xl font-bold text-white mt-10 mb-4">1. Acceptance of Terms</h3>
                        <p className="text-blue-100 mb-6 leading-relaxed">
                            By accessing or using BioDockify, you agree to be bound by these Terms of Service.
                        </p>

                        <h3 className="text-2xl font-bold text-white mt-10 mb-4">2. Use of Service</h3>
                        <p className="text-blue-100 mb-6 leading-relaxed">
                            You agree to use BioDockify only for lawful purposes and in accordance with these Terms. You are responsible for all activities that occur under your account.
                        </p>

                        <h3 className="text-2xl font-bold text-white mt-10 mb-4">3. Intellectual Property</h3>
                        <p className="text-blue-100 mb-6 leading-relaxed">
                            You retain all rights to the data and files you upload. BioDockify claims no ownership over your research data.
                        </p>

                        <h3 className="text-2xl font-bold text-white mt-10 mb-4">4. Limitation of Liability</h3>
                        <p className="text-blue-100 mb-6 leading-relaxed">
                            BioDockify is provided "as is" without warranties of any kind. We are not liable for any damages arising from your use of the service.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
