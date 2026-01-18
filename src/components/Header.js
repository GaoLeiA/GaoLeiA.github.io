'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Github, Home, Info, Cpu, Code, Terminal } from 'lucide-react';
import styles from './Header.module.css'; // We will create this

export default function Header() {
    const pathname = usePathname();
    const isActive = (path) => pathname === path;
    const isCategoryActive = (cat) => pathname?.includes(`/category/${cat}`);

    return (
        <div className={styles.wrapper}>
            <div className="container">
                <header className={styles.header}>
                    {/* Logo */}
                    <Link href="/" className={styles.logoGroup}>
                        <div className={styles.avatar}>
                            <img src="https://avatars.githubusercontent.com/u/GaoLeiA" alt="Lei" />
                        </div>
                        <div className={styles.siteInfo}>
                            <h1 className={styles.title}>Lei's Blog</h1>
                            <span className={styles.subtitle}>AI · Coding · System</span>
                        </div>
                    </Link>

                    {/* Navigation */}
                    <nav className={styles.nav}>
                        <Link href="/" className={`${styles.navLink} ${isActive('/') ? styles.active : ''}`}>
                            <Home size={16} /> 首页
                        </Link>
                        <Link href="/about" className={`${styles.navLink} ${isActive('/about') ? styles.active : ''}`}>
                            <Info size={16} /> 关于
                        </Link>
                        <a href="https://github.com/GaoLeiA" target="_blank" className={styles.navLink}>
                            <Github size={16} /> GitHub
                        </a>

                        <div className={styles.divider}></div>

                        <Link href="/category/ai" className={`${styles.catLink} ${styles.ai} ${isCategoryActive('ai') ? styles.activeCat : ''}`}>
                            <Cpu size={14} /> AI
                        </Link>
                        <Link href="/category/coding" className={`${styles.catLink} ${styles.coding} ${isCategoryActive('coding') ? styles.activeCat : ''}`}>
                            <Code size={14} /> Coding
                        </Link>
                        <Link href="/category/system" className={`${styles.catLink} ${styles.system} ${isCategoryActive('system') ? styles.activeCat : ''}`}>
                            <Terminal size={14} /> 系统
                        </Link>
                    </nav>
                </header>
            </div>
        </div>
    );
}
