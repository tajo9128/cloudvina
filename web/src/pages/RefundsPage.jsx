import React from 'react'
import SEOHelmet from '../components/SEOHelmet'

export default function RefundsPage() {
    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <SEOHelmet
                title="Refund & Cancellation Policy | BioDockify"
                description="Our refund and cancellation policy for paid services."
            />
            <div className="container mx-auto px-4 max-w-4xl">
                <div className="bg-white p-8 md:p-12 rounded-2xl shadow-sm border border-slate-200">
                    <h1 className="text-3xl font-bold text-slate-900 mb-8">Refund & Cancellation Policy</h1>

                    <div className="prose prose-slate max-w-none text-slate-600">
                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Cancellation Policy</h2>
                            <p className="mb-4">
                                BioDockify offers a flexible cancellation policy for our subscription plans.
                                You may cancel your subscription at any time directly from your account dashboard or by contacting our support team.
                            </p>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>
                                    <strong>Immediate Cancellation:</strong> If you cancel your subscription, your account will remain active until the end of your current billing cycle.
                                </li>
                                <li>
                                    <strong>No Penalty:</strong> There are no cancellation fees.
                                </li>
                                <li>
                                    <strong>Renewal:</strong> To avoid being charged for the next billing cycle, please cancel at least 24 hours before your renewal date.
                                </li>
                            </ul>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Refund Policy</h2>
                            <p className="mb-4">
                                We strive to ensure satisfaction with our molecular docking and analysis services.
                                Our refund policy is designed to be fair and transparent (compliant with Instamojo & Payment Partner guidelines).
                            </p>

                            <h3 className="text-lg font-bold text-slate-800 mt-6 mb-3">1. Eligibility for Refunds</h3>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>
                                    <strong>Technical Failure:</strong> If a docking job fails due to our system error and credits were deducted, we will automatically refund the credits. If a paid "top-up" was used for failed jobs, a monetary refund can be requested within 7 days.
                                </li>
                                <li>
                                    <strong>Accidental Charge:</strong> If you were charged due to a technical error (e.g., double billing), a full refund will be issued immediately upon verification.
                                </li>
                                <li>
                                    <strong>Service Unavailability:</strong> If the platform is down for more than 24 consecutive hours, you may request a pro-rated refund for that month.
                                </li>
                            </ul>

                            <h3 className="text-lg font-bold text-slate-800 mt-6 mb-3">2. Non-Refundable Circumstances</h3>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>
                                    Refunds are not granted for "change of mind" after significant usage of the service (e.g., running multiple successful docking jobs).
                                </li>
                                <li>
                                    Partial months of a subscription are generally not refunded upon cancellation (service continues until the end of the billing period).
                                </li>
                            </ul>

                            <h3 className="text-lg font-bold text-slate-800 mt-6 mb-3">3. Processing Timeframe</h3>
                            <p>
                                Refund requests are processed within <strong>5-7 business days</strong>.
                                Once approved, the funds will be credited back to your original payment method (bank account, credit card, or UPI) via our payment partner (Instamojo/Stripe).
                            </p>
                        </section>

                        <section className="mb-8">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Contact Us</h2>
                            <p className="mb-4">
                                If you have any questions about our Refunds & Cancellation Policy, please contact us:
                            </p>
                            <ul className="list-none space-y-2">
                                <li>
                                    <strong>Email:</strong> support@biodockify.com
                                </li>
                                <li>
                                    <strong>Phone:</strong> +91-9667362181
                                </li>
                            </ul>
                        </section>

                        <div className="mt-8 pt-8 border-t border-slate-100 text-sm text-slate-500">
                            <p>Last updated: December 2025</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
