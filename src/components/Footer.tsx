/**
 * INDILA · Footer
 * Deep navy surface. Single disciplined lockup with nav + admissions block.
 * Purposely quiet. This is an education brand, not a SaaS.
 */

import { Link } from 'react-router-dom';
import { IndilaLogo } from './IndilaMark';

const EXPLORE = [
  { label: 'For Athletes',     href: '#for-athletes' },
  { label: 'How It Works',     href: '#how-it-works' },
  { label: 'Coaches & Clubs',  href: '#coaches' },
  { label: 'Funding',          href: '#funding' },
  { label: 'FAQ',              href: '#faq' },
];

const CONNECT = [
  { label: 'Request Information', href: '#request-info' },
  { label: 'Schedule a Call',     href: '#admissions' },
  // TODO(admissions): replace with the real admissions email when available.
  { label: 'admissions@indila.org', href: 'mailto:admissions@indila.org' },
];

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer style={{ background: '#0E2544', color: '#FAF7F0' }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '0 clamp(24px, 5vw, 72px)' }}>

        {/* Top: lockup + nav + CTA */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(12, 1fr)',
          gap: '48px',
          paddingTop: 'clamp(64px, 8vw, 104px)',
          paddingBottom: 'clamp(44px, 5vw, 72px)',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
        }} className="footer-grid">

          {/* Brand */}
          <div style={{ gridColumn: 'span 5' }}>
            <IndilaLogo onDark size={42} />

            <p style={{
              fontSize: '14px',
              lineHeight: 1.8,
              color: 'rgba(250,247,240,0.55)',
              maxWidth: '400px',
              marginTop: '28px',
              marginBottom: '32px',
            }}>
              An accredited K–12 academy built for Indiana families whose student's
              schedule no longer fits the traditional day. Flexibility with structure:
              serious academics, real teachers, real progress.
            </p>

            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'rgba(250,247,240,0.35)', marginBottom: '6px' }}>
                Admissions
              </div>
              <a href="mailto:admissions@indila.org" style={{ fontSize: '14px', color: 'rgba(250,247,240,0.75)', textDecoration: 'none' }}>
                admissions@indila.org
              </a>
            </div>
          </div>

          {/* Explore */}
          <div style={{ gridColumn: 'span 3' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'rgba(250,247,240,0.38)', marginBottom: '22px' }}>
              Explore
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '13px' }}>
              {EXPLORE.map(l => (
                <a key={l.href} href={l.href} style={{ fontSize: '13.5px', color: 'rgba(250,247,240,0.65)', textDecoration: 'none', transition: 'color 0.2s' }}
                  onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = '#D4A029')}
                  onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = 'rgba(250,247,240,0.65)')}
                >
                  {l.label}
                </a>
              ))}
            </div>
          </div>

          {/* Connect */}
          <div style={{ gridColumn: 'span 4' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'rgba(250,247,240,0.38)', marginBottom: '22px' }}>
              Connect
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '13px', marginBottom: '28px' }}>
              {CONNECT.map(l => (
                <a key={l.label} href={l.href} style={{ fontSize: '13.5px', color: 'rgba(250,247,240,0.65)', textDecoration: 'none', transition: 'color 0.2s' }}
                  onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = '#D4A029')}
                  onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = 'rgba(250,247,240,0.65)')}
                >
                  {l.label}
                </a>
              ))}
            </div>

            <a
              href="#request-info"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '13px 22px',
                background: '#D4A029',
                color: '#15355B',
                fontSize: '11px',
                fontWeight: 700,
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                textDecoration: 'none',
                transition: 'background 0.2s',
              }}
              onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = '#E0AE31')}
              onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = '#D4A029')}
            >
              Request Information
            </a>
          </div>
        </div>

        {/* Bottom bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
          paddingTop: '24px',
          paddingBottom: '36px',
        }}>
          <span style={{ fontSize: '11.5px', color: 'rgba(250,247,240,0.35)' }}>
            © {year} Indiana Individualized Learning Academies. All rights reserved.
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap' }}>
            <Link to="/privacy" style={{ fontSize: '11.5px', color: 'rgba(250,247,240,0.35)', textDecoration: 'none' }}>Privacy</Link>
            <Link to="/accessibility" style={{ fontSize: '11.5px', color: 'rgba(250,247,240,0.35)', textDecoration: 'none' }}>Accessibility</Link>
            <span style={{ fontSize: '11.5px', color: 'rgba(250,247,240,0.35)' }}>Indiana · USA</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
