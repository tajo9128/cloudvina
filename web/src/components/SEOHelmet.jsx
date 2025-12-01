import { Helmet } from 'react-helmet-async'

export default function SEOHelmet({
    title = 'BioDockify - Free Molecular Docking Online',
    description = 'Run AutoDock Vina molecular docking online free. Cloud-based drug discovery platform for students & researchers.',
    keywords = 'molecular docking online free, autodock vina online, free molecular docking',
    canonical = 'https://biodockify.com',
    ogImage = 'https://biodockify.com/assets/images/og-image.png',
    schema = null
}) {
    return (
        <Helmet>
            {/* Basic Meta Tags */}
            <title>{title}</title>
            <meta name="description" content={description} />
            <meta name="keywords" content={keywords} />
            <link rel="canonical" href={canonical} />

            {/* Open Graph / Facebook */}
            <meta property="og:type" content="website" />
            <meta property="og:url" content={canonical} />
            <meta property="og:title" content={title} />
            <meta property="og:description" content={description} />
            <meta property="og:image" content={ogImage} />
            <meta property="og:site_name" content="BioDockify" />

            {/* Twitter Card */}
            <meta name="twitter:card" content="summary_large_image" />
            <meta name="twitter:url" content={canonical} />
            <meta name="twitter:title" content={title} />
            <meta name="twitter:description" content={description} />
            <meta name="twitter:image" content={ogImage} />

            {/* Additional Meta */}
            <meta name="robots" content="index, follow" />
            <meta name="language" content="English" />
            <meta name="author" content="BioDockify" />

            {/* Schema.org Structured Data */}
            {schema && (
                <script type="application/ld+json">
                    {JSON.stringify(schema)}
                </script>
            )}
        </Helmet>
    )
}
