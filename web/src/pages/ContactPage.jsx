import { useState } from 'react'

export default function ContactPage() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        subject: '',
        message: ''
    })

    const handleSubmit = (e) => {
        e.preventDefault()
        // For MVP, we'll just open the mail client with the filled details
        const subject = encodeURIComponent(`[BioDockify Inquiry] ${formData.subject}`)
        const body = encodeURIComponent(`Name: ${formData.name}\nEmail: ${formData.email}\n\nMessage:\n${formData.message}`)
        window.location.href = `mailto:biodockify@hotmail.com?subject=${subject}&body=${body}`
    }

    return (
        <div className="min-h-screen bg-slate-50 pt-32 pb-20">
            <div className="container mx-auto px-4">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-16">
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6 tracking-tight">Contact Us</h1>
                        <p className="text-xl text-slate-600 max-w-2xl mx-auto">
                            Have questions about BioDockify? Need help with your docking jobs?
                            We're here to help.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-12">
                        {/* Contact Info */}
                        <div className="space-y-8">
                            <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                                <h3 className="text-xl font-bold text-slate-900 mb-6">Get in Touch</h3>
                                <div className="space-y-6">
                                    <div className="flex items-start">
                                        <div className="w-10 h-10 rounded-lg bg-primary-50 text-primary-600 flex items-center justify-center text-xl mr-4 flex-shrink-0">📧</div>
                                        <div>
                                            <p className="font-bold text-slate-900">Email</p>
                                            <a href="mailto:biodockify@hotmail.com" className="text-primary-600 hover:text-primary-700 font-medium transition-colors">
                                                biodockify@hotmail.com
                                            </a>
                                            <p className="text-sm text-slate-500 mt-1">
                                                We usually respond within 24 hours.
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="w-10 h-10 rounded-lg bg-primary-50 text-primary-600 flex items-center justify-center text-xl mr-4 flex-shrink-0">📍</div>
                                        <div>
                                            <p className="font-bold text-slate-900">Location</p>
                                            <p className="text-slate-600">
                                                7-887/C, Subhah Nagar, Jeedimetla,
                                                <br />Bala Nagar, Hyderabad, 500055
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="w-10 h-10 rounded-lg bg-primary-50 text-primary-600 flex items-center justify-center text-xl mr-4 flex-shrink-0">📱</div>
                                        <div>
                                            <p className="font-bold text-slate-900">Phone</p>
                                            <a href="tel:+919700987475" className="text-primary-600 hover:text-primary-700 font-medium transition-colors">
                                                +91 9700987475
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-primary-600 to-secondary-600 rounded-2xl shadow-xl shadow-primary-900/20 p-8 text-white relative overflow-hidden">
                                <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16 blur-2xl"></div>
                                <h3 className="text-xl font-bold mb-4 relative z-10">Support for Students</h3>
                                <p className="mb-6 opacity-90 relative z-10 text-primary-50 leading-relaxed">
                                    Are you a student or researcher from an Indian university?
                                    We offer special support and extended credits for academic projects.
                                </p>
                                <div className="font-bold relative z-10 bg-white/20 inline-block px-4 py-2 rounded-lg backdrop-blur-sm text-sm">
                                    Mention your institution!
                                </div>
                            </div>
                        </div>

                        {/* Contact Form */}
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">
                                        Your Name
                                    </label>
                                    <input
                                        type="text"
                                        required
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition outline-none text-slate-900"
                                        placeholder="Dr. Rajesh Kumar"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">
                                        Email Address
                                    </label>
                                    <input
                                        type="email"
                                        required
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition outline-none text-slate-900"
                                        placeholder="rajesh@iit.ac.in"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">
                                        Subject
                                    </label>
                                    <select
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition outline-none text-slate-900"
                                        value={formData.subject}
                                        onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                                    >
                                        <option value="" disabled>Select a topic</option>
                                        <option value="General Inquiry">General Inquiry</option>
                                        <option value="Technical Support">Technical Support</option>
                                        <option value="Billing/Credits">Billing & Credits</option>
                                        <option value="Academic Partnership">Academic Partnership</option>
                                        <option value="Bug Report">Bug Report</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">
                                        Message
                                    </label>
                                    <textarea
                                        required
                                        rows="4"
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition outline-none text-slate-900"
                                        placeholder="How can we help you?"
                                        value={formData.message}
                                        onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                    ></textarea>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full btn-primary py-3 rounded-xl font-bold text-lg shadow-lg shadow-primary-600/20"
                                >
                                    Send Message
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
