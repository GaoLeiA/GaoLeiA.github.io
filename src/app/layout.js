import './globals.css';
import Header from '@/components/Header';
import { Inter, JetBrains_Mono } from 'next/font/google';

const inter = Inter({ subsets: ['latin'], variable: '--font-primary', display: 'swap' });
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono', display: 'swap' });

export const metadata = {
  title: "Lei's Blog",
  description: 'AI · Coding · System',
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN" className={`${inter.variable} ${mono.variable}`}>
      <body>
        <Header />
        <main className="container" style={{ padding: '2rem 1.5rem', minHeight: 'calc(100vh - 200px)' }}>
          {children}
        </main>
        <footer style={{
          borderTop: '1px solid var(--dark-border)',
          marginTop: '3rem',
          padding: '2rem 0',
          textAlign: 'center',
          color: 'var(--text-muted)'
        }}>
          <p>© {new Date().getFullYear()} Lei's Blog · Built with Next.js</p>
        </footer>
      </body>
    </html>
  );
}
