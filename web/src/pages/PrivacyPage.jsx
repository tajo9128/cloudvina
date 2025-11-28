export default function PrivacyPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
            <div className="container mx-auto px-4 py-12">
                <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-sm p-8 border border-gray-100">
                    <h1 className="text-3xl font-bold text-gray-900 mb-6">Privacy Policy</h1>
                    <div className="prose prose-purple max-w-none text-gray-600">
                        <p className="mb-4">Last updated: November 23, 2025</p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">1. Information We Collect</h3>
                        <p className="mb-4">
                            We collect information you provide directly to us, such as when you create an account, submit a job, or contact us. This may include your name, email address, phone number, and institutional affiliation.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">2. How We Use Your Information</h3>
                        <p className="mb-4">
                            We use your information to provide, maintain, and improve our services, process your docking jobs, and communicate with you.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">3. Data Security</h3>
                        <p className="mb-4">
                            We implement appropriate technical and organizational measures to protect your data. Your molecular data (receptors and ligands) is processed in secure, isolated containers and is deleted from our processing servers after job completion.
                        </p>

                        <h3 className="text-xl font-bold text-gray-900 mt-8 mb-4">4. Contact Us</h3>
                        <p className="mb-4">
                            If you have any questions about this Privacy Policy, please contact us at BioDockify2025@gmail.com.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
