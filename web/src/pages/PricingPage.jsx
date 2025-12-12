import { CheckCircle, Zap, Shield, HelpCircle } from 'lucide-react'
import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'

export default function PricingPage() {
    const tiers = [
        {
            name: "Free",
            price: "₹0",
            period: "/forever",
            description: "Essential tools for students and casual docking.",
            features: [
                "3 Docking Runs / day",
                "Standard Speed",
                "Public Support",
                "Basic Results"
            ],
            cta: "Start Free",
            ctaLink: "/auth/signup",
            highlighted: false
        },
        {
            name: "Pro Research",
            price: "₹100",
            period: "/bundle",
            description: "For serious researchers requiring bulk processing.",
            features: [
                "50 Docking Jobs (Individual)",
                "OR 10 x 5 Batch Jobs",
                "Standard Rate: ₹2 / job",
                "Priority Queue",
                "Private Storage",
                "Email Support"
            ],
            cta: "Buy Bundle",
            ctaLink: "/auth/signup?plan=pro",
            highlighted: true
        },
        {
            name: "Premium Bundle",
            price: "₹500",
            period: "/bundle",
            description: "Complete pipeline access for deep drug discovery.",
            features: [
                "100 Docking Jobs (Individual)",
                "OR 10 x 10 Batch Jobs",
                "Full 7-Phase Pipeline (1 Drug)",
                "Includes MD & ADMET Analysis",
                "Priority GPU Access",
                "Results Export (PDF)"
            ],
            cta: "Buy Premium",
            ctaLink: "/auth/signup?plan=premium",
            highlighted: false
        }
    ]

    return (
        <div className="bg-slate-50 min-h-screen pt-24 pb-24">
            <SEOHelmet
                title="Pricing Plans | BioDockify"
                description="Affordable molecular docking at ₹2/job. Free tier included."
                keywords="molecular docking pricing, cheap docking, indian bioinformatics pricing"
            />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Hero Image Section */}
                <div className="mb-16 rounded-3xl overflow-hidden shadow-2xl relative h-64 md:h-80 w-full">
                    <img
                        src="https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=2000"
                        alt="Scientific Research and Pricing"
                        className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-r from-slate-900/90 to-slate-900/50 flex items-center justify-center">
                        <div className="text-center px-4">
                            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">Simple, Transparent Pricing</h1>
                            <p className="text-xl text-slate-200 max-w-2xl mx-auto">
                                High-performance computing made affordable. <br />
                                <span className="text-green-400 font-bold">₹2 per docking job</span> with no hidden fees.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="text-center max-w-3xl mx-auto mb-12">
                    {/* Removed previous text since it's now in the header image overlay */}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-24">
                    {tiers.map((tier, i) => (
                        <div key={i} className={`relative bg-white rounded-2xl shadow-lg p-8 border ${tier.highlighted ? 'border-primary-500 ring-4 ring-primary-500/10' : 'border-slate-100'}`}>
                            {tier.highlighted && (
                                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-primary-600 text-white px-4 py-1 rounded-full text-sm font-bold uppercase tracking-wide">
                                    Best Value
                                </div>
                            )}
                            <h3 className="text-xl font-bold text-slate-900 mb-2">{tier.name}</h3>
                            <div className="flex items-baseline mb-4">
                                <span className="text-4xl font-bold text-slate-900">{tier.price}</span>
                                <span className="text-slate-500 ml-1 text-sm font-medium">{tier.period}</span>
                            </div>
                            <p className="text-slate-600 mb-8 h-12 text-sm">{tier.description}</p>

                            <ul className="space-y-4 mb-8">
                                {tier.features.map((feature, j) => (
                                    <li key={j} className="flex items-start">
                                        <CheckCircle className="text-emerald-500 mr-3 shrink-0" size={20} />
                                        <span className="text-slate-700 text-sm font-medium">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            <Link
                                to={tier.ctaLink}
                                className={`block w-full py-4 rounded-xl text-center font-bold transition-all ${tier.highlighted
                                    ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-primary-600/30'
                                    : 'bg-slate-50 text-slate-900 hover:bg-slate-100 border border-slate-200'
                                    }`}
                            >
                                {tier.cta}
                            </Link>
                        </div>
                    ))}
                </div>

                {/* FAQ Section */}
                <div className="max-w-3xl mx-auto">
                    <h2 className="text-2xl font-bold text-slate-900 mb-8 text-center">Frequently Asked Questions</h2>
                    <div className="space-y-6">
                        {[
                            { q: "How does the ₹100 bundle work?", a: "You pay ₹100 and get 50 docking credits. You can use these for 50 individual jobs or 10 batch jobs (each batch containing 5 ligands)." },
                            { q: "Is the Free plan really free?", a: "Yes. You get 3 jobs every single day for free. No credit card required." },
                            { q: "Can I buy more credits?", a: "Absolutley. Once you use your bundle, you can simply purchase another one or contact us for custom bulk rates." }
                        ].map((faq, i) => (
                            <div key={i} className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                                <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                                    <HelpCircle size={18} className="text-primary-500" /> {faq.q}
                                </h4>
                                <p className="text-slate-600 text-sm ml-6">{faq.a}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}
