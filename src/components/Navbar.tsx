/**
 * INDILA · Navbar
 *
 * Fixed 80px. Paper surface on light pages, transparent over hero.
 * Left: IndilaLogo lockup (brand-faithful mark + wordmark + tagline).
 * Right: anchor links + primary CTA. The CTA uses INDILA gold on ink
 * so it reads as the single most important action anywhere on the page.
 */

import { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { IndilaLogo } from './IndilaMark';

const REQUEST_HREF = '#request-info';

const NAV_LINKS = [
  { label: 'For Athletes',     href: '#for-athletes' },
  { label: 'How It Works',     href: '#how-it-works' },
  { label: 'Coaches & Clubs',  href: '#coaches' },
  { label: 'FAQ',              href: '#faq' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const ticking = useRef(false);

  const isLanding = location.pathname === '/' || location.pathname === '/athletics';

  useEffect(() => {
    function onScroll() {
      if (!ticking.current) {
        requestAnimationFrame(() => {
          setScrolled(window.scrollY > 56);
          ticking.current = false;
        });
        ticking.current = true;
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => { setMobileOpen(false); }, [location]);

  const transparent = isLanding && !scrolled;

  return (
    <>
      <nav style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        height: '80px',
        display: 'flex',
        alignItems: 'center',
        transition: 'background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease',
        background:   transparent ? 'transparent'                 : '#FAF7F0',
        borderBottom: transparent ? '1px solid transparent'       : '1px solid #E9E3D4',
        boxShadow:    transparent ? 'none'                        : '0 1px 14px rgba(21,53,91,0.05)',
      }}>
        <div style={{
          maxWidth: '1440px',
          width: '100%',
          margin: '0 auto',
          padding: '0 clamp(20px, 5vw, 72px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>

          {/* Logo lockup */}
          <Link to="/" style={{ textDecoration: 'none', flexShrink: 0 }}>
            <IndilaLogo onDark={transparent} size={38} />
          </Link>

          {/* Desktop links */}
          <div className="hidden lg:flex" style={{ alignItems: 'center', gap: '36px' }}>
            {NAV_LINKS.map(link => (
              <a
                key={link.href}
                href={link.href}
                style={{
                  fontSize: '13px',
                  fontWeight: 500,
                  letterSpacing: '0.01em',
                  textDecoration: 'none',
                  color: transparent ? 'rgba(250,247,240,0.80)' : '#404C5E',
                  transition: 'color 0.2s ease',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.color = transparent ? '#FAF7F0' : '#15355B';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.color = transparent ? 'rgba(250,247,240,0.80)' : '#404C5E';
                }}
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* CTA + mobile */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <a
              href={REQUEST_HREF}
              className="hidden lg:inline-flex"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 20px',
                fontSize: '12.5px',
                fontWeight: 600,
                letterSpacing: '0.005em',
                textTransform: 'none',
                textDecoration: 'none',
                whiteSpace: 'nowrap',
                background: transparent ? '#D4A029' : '#15355B',
                color: transparent ? '#15355B' : '#FAF7F0',
                border: '1px solid ' + (transparent ? '#D4A029' : '#15355B'),
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLElement;
                if (transparent) { el.style.background = '#E0AE31'; el.style.borderColor = '#E0AE31'; }
                else { el.style.background = '#1C436E'; el.style.borderColor = '#1C436E'; }
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLElement;
                el.style.background = transparent ? '#D4A029' : '#15355B';
                el.style.borderColor = transparent ? '#D4A029' : '#15355B';
              }}
            >
              Request Information
            </a>

            <button
              onClick={() => setMobileOpen(v => !v)}
              aria-label="Toggle navigation"
              className="lg:hidden"
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '6px',
                color: transparent ? '#FAF7F0' : '#15355B',
                display: 'flex',
                alignItems: 'center',
                transition: 'color 0.4s ease',
              }}
            >
              {mobileOpen ? <X size={22} strokeWidth={1.8} /> : <Menu size={22} strokeWidth={1.8} />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div style={{
          position: 'fixed',
          top: '80px',
          left: 0,
          right: 0,
          zIndex: 99,
          background: '#FAF7F0',
          borderBottom: '1px solid #E9E3D4',
          boxShadow: '0 8px 40px rgba(21,53,91,0.08)',
        }}>
          <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '16px clamp(20px, 5vw, 40px) 24px' }}>
            {NAV_LINKS.map(link => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                style={{
                  display: 'block',
                  padding: '14px 0',
                  fontSize: '15px',
                  fontWeight: 500,
                  color: '#404C5E',
                  textDecoration: 'none',
                  borderBottom: '1px solid #E9E3D4',
                }}
              >
                {link.label}
              </a>
            ))}
            <a
              href={REQUEST_HREF}
              onClick={() => setMobileOpen(false)}
              style={{
                display: 'block',
                marginTop: '18px',
                padding: '15px',
                background: '#15355B',
                color: '#FAF7F0',
                textAlign: 'center',
                fontSize: '13px',
                fontWeight: 600,
                letterSpacing: '0.005em',
                textTransform: 'none',
                textDecoration: 'none',
              }}
            >
              Request Information
            </a>
          </div>
        </div>
      )}
    </>
  );
}
