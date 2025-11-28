export default function TermsPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <div className="container mx-auto px-4 py-12">
                <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-sm p-8 border border-gray-100">
                    <h1 className="text-3xl font-bold text-gray-900 mb-6">Terms of Service</h1>
                    <div className="prose prose-purple max-w-none text-gray-600">
                        <p className="mb-4">Last updated: November 23, 2025</p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">1. Acceptance of Terms</h3>
                        <p className="mb-4">
                            By accessing or using BioDockify, you agree to be bound by these Terms of Service.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">2. Use of Service</h3>
                        <p className="mb-4">
                            You agree to use BioDockify only for lawful purposes and in accordance with these Terms. You are responsible for all activities that occur under your account.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">3. Intellectual Property</h3>
                        <p className="mb-4">
                            You retain all rights to the data and files you upload. BioDockify claims no ownership over your research data.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">4. Limitation of Liability</h3>
                        <p className="mb-4">
                            BioDockify is provided "as is" without warranties of any kind. We are not liable for any damages arising from your use of the service.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
