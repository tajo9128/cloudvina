import Header from './Header'
import Footer from './Footer'
import ErrorBoundary from './ErrorBoundary'

export default function Layout({ children }) {
    return (
        <div className="min-h-screen flex flex-col bg-slate-50 font-sans text-slate-900">
            <Header />
            <main className="flex-grow pt-20">
                <ErrorBoundary>
                    {children}
                </ErrorBoundary>
            </main>
            <Footer />
        </div>
    )
}
