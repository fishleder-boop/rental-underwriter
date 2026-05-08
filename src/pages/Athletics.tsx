/**
 * INDILA · Athletics Landing
 *
 * Audience:
 *   1. Parents of serious student-athletes
 *   2. Student-athletes with demanding schedules
 *   3. Athletic associations, clubs, training programs, sports academies
 *
 * Primary goal:    on-page form conversion (Request Information)
 * Secondary goal:  scheduled call with admissions
 *
 * Brand system carried forward from the live www.indila.org site:
 *   · Navy   #15355B  shield, headings, dark surfaces
 *   · Gold   #D4A029  primary accent, rules, CTA
 *   · Paper  #FAF7F0  warm off-white page surface
 *   · Display:  Jost (light 300) as a Futura LT Light analog
 *   · Body:     Helvetica Neue
 *
 * Conversion path:
 *   Hero form (id="request-info") is the canonical lead form. All upper-half
 *   CTAs scroll there. Lower-half CTAs scroll down to the closing compact
 *   form (id="submit-inquiry") to avoid making users jump back to the top.
 */

import { useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowRight, Plus, Minus, Check } from 'lucide-react';
import IndilaMark from '../components/IndilaMark';

// ─── Design tokens ───────────────────────────────────────────────────────────
//   Navy           #15355B  primary brand. Logo, headings, dark surfaces.
//   Navy deep      #0E2544  deeper bg for contrast (footer, CTA)
//   Navy deepest   #091528  final CTA floor
//   Paper          #FAF7F0  warm off-white page background
//   Bone           #F1ECE0  alternating cream
//   Sand           #E5DEC9  stronger cream, used sparingly
//   Gold           #D4A029  primary accent. Rules, CTA, marks.
//   Gold hover     #E0AE31  hover state on gold
//   Gold stroke    #DBA428  decorative line variant
//   Body text      #1A2333  body on light
//   Gray mid       #5A6570  secondary body
//   Gray light     #8D949C  captions
//   Border         #D9D1BD  main dividers
//   Border light   #E9E3D4  soft dividers

// Canonical anchor for every "Request Information" CTA on the page.
// All inquiry CTAs route to the single hero form.
const FORM_ANCHOR_TOP = '#request-info';

// TODO: Enable this section only after approved real testimonials are available.
// Hidden by default so the public-facing page never shows placeholder cards.
const SHOW_TESTIMONIAL_PLACEHOLDERS = false;

// ─── Small shared primitives ─────────────────────────────────────────────────

function Eyebrow({
  children,
  onDark = false,
  tone = 'gold',
}: {
  children: React.ReactNode;
  onDark?: boolean;
  tone?: 'gold' | 'navy';
}) {
  const rule =
    tone === 'gold'
      ? '#D4A029'
      : onDark
      ? 'rgba(250,247,240,0.35)'
      : '#15355B';
  const label = onDark ? 'rgba(250,247,240,0.70)' : '#5A6570';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '22px' }}>
      <div style={{ width: '28px', height: '1.5px', background: rule }} />
      <span className="eyebrow" style={{ color: label }}>{children}</span>
    </div>
  );
}

function PrimaryCTA({
  children,
  href = FORM_ANCHOR_TOP,
  onDark = false,
}: {
  children: React.ReactNode;
  href?: string;
  onDark?: boolean;
}) {
  const bg = onDark ? '#D4A029' : '#15355B';
  const bgHover = onDark ? '#E0AE31' : '#1C436E';
  const fg = onDark ? '#15355B' : '#FAF7F0';
  return (
    <a
      href={href}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '12px',
        padding: '18px 32px',
        background: bg,
        color: fg,
        fontSize: '12px',
        fontWeight: 700,
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        textDecoration: 'none',
        transition: 'background 0.2s ease',
      }}
      onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = bgHover)}
      onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = bg)}
    >
      {children}
      <ArrowRight size={14} strokeWidth={2} />
    </a>
  );
}

function SecondaryCTA({
  children,
  href,
  onDark = false,
}: {
  children: React.ReactNode;
  href: string;
  onDark?: boolean;
}) {
  const fg = onDark ? 'rgba(250,247,240,0.92)' : '#15355B';
  const border = onDark ? 'rgba(250,247,240,0.30)' : '#15355B';
  return (
    <a
      href={href}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '10px',
        padding: '17px 28px',
        background: 'transparent',
        color: fg,
        border: '1px solid ' + border,
        fontSize: '12px',
        fontWeight: 600,
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        textDecoration: 'none',
        transition: 'background 0.2s ease, color 0.2s ease',
      }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        if (onDark) {
          el.style.background = 'rgba(250,247,240,0.06)';
        } else {
          el.style.background = '#15355B';
          el.style.color = '#FAF7F0';
        }
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.background = 'transparent';
        el.style.color = fg;
      }}
    >
      {children}
    </a>
  );
}

function GhostLink({
  children,
  href,
  onDark = false,
}: {
  children: React.ReactNode;
  href: string;
  onDark?: boolean;
}) {
  const color = onDark ? 'rgba(250,247,240,0.85)' : '#15355B';
  const border = onDark ? 'rgba(250,247,240,0.30)' : '#15355B';
  return (
    <a
      href={href}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '12px',
        fontWeight: 600,
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        textDecoration: 'none',
        color,
        borderBottom: '1px solid ' + border,
        paddingBottom: '4px',
        transition: 'opacity 0.2s ease',
      }}
      onMouseEnter={e => ((e.currentTarget as HTMLElement).style.opacity = '0.65')}
      onMouseLeave={e => ((e.currentTarget as HTMLElement).style.opacity = '1')}
    >
      {children}
      <ArrowRight size={12} />
    </a>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// INQUIRY FORM
// One canonical form component, used in two places:
//   · variant="hero"     ... the premium right-hand panel in section 01
//   · variant="compact"  ... the closing form in section 11
//
// TODO(form-backend): wire onSubmit to the real lead destination (Wix Forms
// endpoint, HubSpot, ActiveCampaign, etc). Until that integration lands the
// form runs locally: preventDefault, log payload, and present a confirmation
// state so the page is never a dead form.
// ═════════════════════════════════════════════════════════════════════════════

const GRADE_OPTIONS = [
  'Kindergarten',
  '1st grade',
  '2nd grade',
  '3rd grade',
  '4th grade',
  '5th grade',
  '6th grade',
  '7th grade',
  '8th grade',
  '9th grade',
  '10th grade',
  '11th grade',
  '12th grade',
  'Not sure yet',
];

const ROLE_OPTIONS = [
  'Parent',
  'Coach',
  'Club Director',
  'Athletic Association Leader',
  'Other',
];

const INTEREST_OPTIONS = [
  'Full-time school',
  'Part-time courses',
  'Flexible schedule',
  'Information for families',
  'Partnership conversation',
];

type InquiryFormProps = {
  id: string;
  variant: 'hero' | 'compact';
  headline?: string;
  subtext?: string;
  buttonLabel?: string;
};

function InquiryForm({
  id,
  variant,
  headline = 'Request Information',
  subtext = 'For parents, coaches, clubs, and athletic associations looking for a flexible accredited school option.',
  buttonLabel = 'Request Information',
}: InquiryFormProps) {
  const [submitted, setSubmitted] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    // TODO(form-backend): replace this local handler with the real submission.
    // For now we capture the payload to the console so QA can confirm fields
    // serialize correctly before the integration is wired up.
    if (formRef.current) {
      const data = Object.fromEntries(new FormData(formRef.current).entries());
      // eslint-disable-next-line no-console
      console.log('[INDILA inquiry]', data);
    }
    setSubmitted(true);
  }

  const isHero = variant === 'hero';
  const surface = '#FAF7F0';
  const fieldBg = '#FFFFFF';
  const fieldBorder = '#D9D1BD';
  const fieldFocus = '#15355B';
  const labelColor = '#404C5E';

  if (submitted) {
    return (
      <div
        id={id}
        style={{
          background: surface,
          padding: 'clamp(32px, 4vw, 48px)',
          border: '1px solid ' + fieldBorder,
          borderTop: '3px solid #D4A029',
        }}
      >
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '54px',
          height: '54px',
          background: '#15355B',
          marginBottom: '24px',
        }}>
          <Check size={22} color="#D4A029" strokeWidth={2.5} />
        </div>
        <h3 className="display" style={{
          fontSize: 'clamp(22px, 2.4vw, 30px)',
          color: '#15355B',
          lineHeight: 1.2,
          marginBottom: '14px',
        }}>
          Thank you. We received your inquiry.
        </h3>
        <p style={{ fontSize: '15px', color: '#5A6570', lineHeight: 1.75, marginBottom: '24px' }}>
          A member of the INDILA admissions team will follow up by email to
          answer questions about academics, flexibility, support, and
          scholarship eligibility.
        </p>
      </div>
    );
  }

  return (
    <form
      id={id}
      ref={formRef}
      onSubmit={onSubmit}
      noValidate={false}
      style={{
        background: surface,
        padding: isHero ? 'clamp(22px, 2.4vw, 32px)' : 'clamp(22px, 2.4vw, 34px)',
        border: '1px solid ' + fieldBorder,
        borderTop: '3px solid #D4A029',
      }}
      aria-labelledby={id + '-heading'}
    >
      <div style={{ marginBottom: '18px' }}>
        <span className="eyebrow" style={{ color: '#D4A029', display: 'block', marginBottom: '10px' }}>
          Inquiry Form
        </span>
        <h3
          id={id + '-heading'}
          className="display"
          style={{
            fontSize: 'clamp(20px, 2vw, 26px)',
            color: '#15355B',
            lineHeight: 1.2,
            marginBottom: '8px',
          }}
        >
          {headline}
        </h3>
        <p style={{ fontSize: '13px', color: '#5A6570', lineHeight: 1.55 }}>
          {subtext}
        </p>
      </div>

      <div className="form-grid" style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '11px 12px',
      }}>
        <Field label="First Name"  name="first_name" autoComplete="given-name"  required />
        <Field label="Last Name"   name="last_name"  autoComplete="family-name" required />
        <Field label="Email"       name="email"      type="email" autoComplete="email" required full />
        <Field label="Phone"       name="phone"      type="tel"   autoComplete="tel"   required full />

        <SelectField
          label="Student Grade"
          name="student_grade"
          options={GRADE_OPTIONS}
          placeholder="Select a grade"
          required
        />
        <SelectField
          label="I am a"
          name="role"
          options={ROLE_OPTIONS}
          placeholder="Select one"
          required
        />
        <Field
          label="Sport or Athletic Program"
          name="sport"
          placeholder="e.g. Soccer, Swimming, Gymnastics"
          full
        />
        <TextareaField
          label="Message (optional)"
          name="message"
          rows={isHero ? 2 : 3}
          placeholder="Tell us about your student's schedule, goals, or any questions."
          full
        />
      </div>

      <button
        type="submit"
        style={{
          marginTop: '18px',
          width: '100%',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          padding: '15px 28px',
          background: '#15355B',
          color: '#FAF7F0',
          fontSize: '12px',
          fontWeight: 700,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          border: 'none',
          cursor: 'pointer',
          fontFamily: 'inherit',
          transition: 'background 0.2s ease',
        }}
        onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = '#1C436E')}
        onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = '#15355B')}
      >
        {buttonLabel}
        <ArrowRight size={14} strokeWidth={2} />
      </button>

      <p style={{
        marginTop: '12px',
        fontSize: '12px',
        color: '#8D949C',
        lineHeight: 1.55,
      }}>
        No commitment required. Our admissions team will follow up with next steps.
      </p>

      <style>{`
        .form-grid input, .form-grid select, .form-grid textarea {
          width: 100%;
          padding: 10px 12px;
          background: ${fieldBg};
          border: 1px solid ${fieldBorder};
          border-radius: 0;
          font-family: inherit;
          font-size: 14px;
          color: #1A2333;
          transition: border-color 0.18s ease, box-shadow 0.18s ease;
          box-sizing: border-box;
          appearance: none;
          -webkit-appearance: none;
        }
        .form-grid select {
          background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1 L6 6 L11 1' stroke='%2315355B' stroke-width='1.6' fill='none' stroke-linecap='round'/></svg>");
          background-repeat: no-repeat;
          background-position: right 12px center;
          padding-right: 32px;
        }
        .form-grid textarea {
          resize: vertical;
          min-height: 72px;
          line-height: 1.5;
        }
        .form-grid input::placeholder, .form-grid textarea::placeholder { color: #A8AFB7; }
        .form-grid input:focus, .form-grid select:focus, .form-grid textarea:focus {
          outline: none;
          border-color: ${fieldFocus};
          box-shadow: 0 0 0 3px rgba(21,53,91,0.12);
        }
        .form-label {
          display: block;
          font-size: 10.5px;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: ${labelColor};
          margin-bottom: 5px;
        }
        @media (max-width: 640px) {
          .form-grid { grid-template-columns: 1fr !important; }
          .form-grid .field-full { grid-column: auto !important; }
        }
      `}</style>
    </form>
  );
}

function Field({
  label, name, type = 'text', required = false, autoComplete, placeholder, full = false,
}: {
  label: string;
  name: string;
  type?: string;
  required?: boolean;
  autoComplete?: string;
  placeholder?: string;
  full?: boolean;
}) {
  const id = 'fld-' + name;
  return (
    <div className={full ? 'field-full' : ''} style={{ gridColumn: full ? 'span 2' : 'auto' }}>
      <label htmlFor={id} className="form-label">
        {label}{required && <span style={{ color: '#9C2426', marginLeft: '4px' }} aria-hidden="true">*</span>}
      </label>
      <input
        id={id}
        name={name}
        type={type}
        required={required}
        autoComplete={autoComplete}
        placeholder={placeholder}
        aria-required={required || undefined}
      />
    </div>
  );
}

function SelectField({
  label, name, options, required = false, placeholder, full = false,
}: {
  label: string;
  name: string;
  options: string[];
  required?: boolean;
  placeholder?: string;
  full?: boolean;
}) {
  const id = 'fld-' + name;
  return (
    <div className={full ? 'field-full' : ''} style={{ gridColumn: full ? 'span 2' : 'auto' }}>
      <label htmlFor={id} className="form-label">
        {label}{required && <span style={{ color: '#9C2426', marginLeft: '4px' }} aria-hidden="true">*</span>}
      </label>
      <select
        id={id}
        name={name}
        required={required}
        defaultValue=""
        aria-required={required || undefined}
      >
        <option value="" disabled hidden>{placeholder ?? 'Select one'}</option>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function TextareaField({
  label, name, rows = 4, required = false, placeholder, full = false,
}: {
  label: string;
  name: string;
  rows?: number;
  required?: boolean;
  placeholder?: string;
  full?: boolean;
}) {
  const id = 'fld-' + name;
  return (
    <div className={full ? 'field-full' : ''} style={{ gridColumn: full ? 'span 2' : 'auto' }}>
      <label htmlFor={id} className="form-label">
        {label}{required && <span style={{ color: '#9C2426', marginLeft: '4px' }} aria-hidden="true">*</span>}
      </label>
      <textarea
        id={id}
        name={name}
        rows={rows}
        required={required}
        placeholder={placeholder}
        aria-required={required || undefined}
      />
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 01 · HERO with embedded inquiry form
// Deep navy field. Left column: brand mark, headline, subheadline, trust
// bullets, primary + secondary CTAs. Right column: paper inquiry form panel.
// On mobile the form stacks below the hero text.
// ═════════════════════════════════════════════════════════════════════════════

const TRUST_BULLETS = [
  'Accredited K–12 online school',
  'Flexible schedule for training and travel',
  'Certified teachers and counselor support',
  'Free for many Indiana families via the Choice Scholarship',
];

function Hero() {
  return (
    <section style={{ position: 'relative', background: '#15355B', color: '#FAF7F0', overflow: 'hidden' }}>
      {/* Hairline gold brand bar */}
      <div aria-hidden="true" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: '#D4A029', zIndex: 2 }} />

      {/* Atmospheric layer */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'radial-gradient(60% 50% at 18% 25%, rgba(212,160,41,0.10) 0%, transparent 65%),' +
            'linear-gradient(180deg, #15355B 0%, #102A4A 100%)',
          pointerEvents: 'none',
        }}
      />

      {/* Soft duotone athletic background image, low opacity, behind text. */}
      {/* TODO(photography): replace with INDILA-owned athletic imagery. */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            "url('https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=1800&q=85')",
          backgroundSize: 'cover',
          backgroundPosition: 'center 30%',
          opacity: 0.18,
          mixBlendMode: 'luminosity',
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'relative',
          maxWidth: '1440px',
          margin: '0 auto',
          padding: 'clamp(120px, 13vw, 168px) clamp(20px, 5vw, 72px) clamp(56px, 7vw, 96px)',
          display: 'grid',
          gridTemplateColumns: 'repeat(12, 1fr)',
          gap: 'clamp(32px, 4vw, 64px)',
          alignItems: 'start',
        }}
        className="hero-grid"
      >

        {/* Text column */}
        <div style={{ gridColumn: 'span 7' }} className="hero-text">

          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '28px' }}>
            <IndilaMark size={32} navy="#FAF7F0" gold="#D4A029" goldStroke="#DBA428" />
            <div>
              <div className="eyebrow" style={{ color: '#D4A029', marginBottom: '4px' }}>
                INDILA &middot; K&ndash;12 Academy
              </div>
              <div style={{ fontSize: '12.5px', color: 'rgba(250,247,240,0.55)', letterSpacing: '0.02em' }}>
                Flexible virtual learning for Indiana student-athletes
              </div>
            </div>
          </div>

          <h1
            className="display"
            style={{
              fontSize: 'clamp(30px, 4.0vw, 52px)',
              color: '#FAF7F0',
              lineHeight: 1.08,
              marginBottom: '22px',
              letterSpacing: '-0.02em',
              maxWidth: '780px',
            }}
          >
            <span className="hero-h1-line" style={{ display: 'block', whiteSpace: 'nowrap' }}>
              A Flexible Accredited School
            </span>
            <span className="hero-h1-line" style={{ display: 'block', whiteSpace: 'nowrap' }}>
              for Serious Student-Athletes
            </span>
          </h1>

          <p style={{
            fontSize: 'clamp(15.5px, 1.35vw, 17.5px)',
            color: 'rgba(250,247,240,0.82)',
            lineHeight: 1.6,
            maxWidth: '580px',
            marginBottom: '14px',
            fontWeight: 300,
          }}>
            For Indiana student-athletes with demanding schedules, INDILA
            provides flexible virtual learning with certified teachers,
            counselor support, and clear visibility for parents.
          </p>

          <p style={{
            fontSize: 'clamp(13.5px, 1.15vw, 15.5px)',
            color: 'rgba(250,247,240,0.74)',
            lineHeight: 1.6,
            maxWidth: '560px',
            marginBottom: '28px',
            fontWeight: 300,
          }}>
            Students should not have to choose between academic progress and
            athletic commitment.
          </p>

          {/* Trust bullets */}
          <ul style={{
            listStyle: 'none',
            display: 'grid',
            gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
            gap: '10px 24px',
            marginBottom: '36px',
            maxWidth: '600px',
          }} className="trust-list">
            {TRUST_BULLETS.map(b => (
              <li key={b} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span aria-hidden="true" style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '20px',
                  height: '20px',
                  background: 'rgba(212,160,41,0.18)',
                  flexShrink: 0,
                }}>
                  <Check size={12} color="#D4A029" strokeWidth={2.5} />
                </span>
                <span style={{ fontSize: '14px', color: 'rgba(250,247,240,0.88)', lineHeight: 1.45 }}>
                  {b}
                </span>
              </li>
            ))}
          </ul>

          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '14px', marginBottom: '20px' }}>
            <PrimaryCTA onDark href={FORM_ANCHOR_TOP}>Request Information</PrimaryCTA>
          </div>

          <p style={{ fontSize: '12.5px', color: 'rgba(250,247,240,0.50)', letterSpacing: '0.02em' }}>
            For families, coaches, clubs, and athletic associations across Indiana.
          </p>
        </div>

        {/* Form column */}
        <div style={{ gridColumn: 'span 5' }} className="hero-form">
          <InquiryForm id="request-info" variant="hero" />
        </div>
      </div>

      <style>{`
        @media (max-width: 980px) {
          .hero-grid { display: block !important; }
          .hero-text { margin-bottom: 40px; }
          .trust-list { grid-template-columns: 1fr !important; }
          .hero-h1-line { white-space: normal !important; }
        }
      `}</style>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 02 · AUDIENCE SPLIT
// Two equal cards. One for parents and student-athletes, one for clubs and
// associations. The page must speak to both audiences clearly, so this is
// the second thing visitors see after the hero.
// ═════════════════════════════════════════════════════════════════════════════

function AudienceSplit() {
  const cards = [
    {
      title: 'Parents and Student-Athletes',
      accent: '#15355B' as const,
      bg: '#FFFFFF' as const,
      copy: 'For families whose training, travel, and competition schedules do not fit a traditional school day, INDILA provides a flexible accredited school option with real structure and support.',
      bullets: [
        'Flexible daily schedule',
        'Teacher-supported coursework',
        'Parent visibility into progress',
        'Counselor and school staff support',
      ],
    },
    {
      title: 'Athletic Associations and Clubs',
      accent: '#D4A029' as const,
      bg: '#F1ECE0' as const,
      copy: 'For clubs, coaches, and athletic associations, INDILA offers a credible academic option to share with families who need more flexibility.',
      bullets: [
        'A resource for families with demanding schedules',
        'A real school option, not just online courses',
        'Support for full-time or flexible learning needs',
        'A credible option for families who need more flexibility',
      ],
    },
  ];

  return (
    <section id="for-athletes" style={{ background: '#FAF7F0' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: 'clamp(48px, 6vw, 76px) clamp(20px, 5vw, 72px)' }}>

        <div style={{ maxWidth: '780px', marginBottom: 'clamp(28px, 3.5vw, 44px)' }}>
          <Eyebrow>Who INDILA is built for</Eyebrow>
          <h2 className="display" style={{
            fontSize: 'clamp(28px, 3.4vw, 44px)',
            color: '#15355B',
            lineHeight: 1.05,
            letterSpacing: '-0.015em',
          }}>
            Built for Families, Clubs, and Athletic Associations
          </h2>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: 'clamp(18px, 2vw, 28px)',
        }} className="audience-grid">
          {cards.map(c => (
            <article
              key={c.title}
              style={{
                background: c.bg,
                borderTop: '3px solid ' + c.accent,
                padding: 'clamp(20px, 2.2vw, 30px)',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <h3 className="display" style={{
                fontSize: 'clamp(19px, 1.9vw, 24px)',
                fontWeight: 400,
                color: '#15355B',
                lineHeight: 1.2,
                marginBottom: '12px',
                letterSpacing: '-0.01em',
              }}>
                {c.title}
              </h3>
              <p style={{ fontSize: '14.5px', color: '#1A2333', lineHeight: 1.7, marginBottom: '16px' }}>
                {c.copy}
              </p>
              <ul style={{ listStyle: 'none', borderTop: '1px solid #D9D1BD', marginTop: 'auto' }}>
                {c.bullets.map(b => (
                  <li key={b} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '10px 0',
                    borderBottom: '1px solid #D9D1BD',
                    fontSize: '14px',
                    color: '#1A2333',
                  }}>
                    <span aria-hidden="true" style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '18px',
                      height: '18px',
                      background: '#15355B',
                      flexShrink: 0,
                    }}>
                      <Check size={11} color="#D4A029" strokeWidth={2.5} />
                    </span>
                    {b}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>

        <div style={{ marginTop: 'clamp(24px, 3vw, 38px)', textAlign: 'center' }}>
          <PrimaryCTA href={FORM_ANCHOR_TOP}>Discuss Supporting Your Athlete Families</PrimaryCTA>
        </div>

        <style>{`
          @media (max-width: 820px) {
            .audience-grid { grid-template-columns: 1fr !important; }
          }
        `}</style>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 03 · THE INDILA APPROACH (problem · solution · credibility, merged)
// Single section that does the work of the old Problem + Solution sections
// AND the old CredibilityAndFit section. Two-column layout: the challenge
// for serious athletes on the left, the credibility points on the right.
// Carries the #funding anchor for the footer/nav scholarship link.
// ═════════════════════════════════════════════════════════════════════════════

function FlexibilityWithStructure() {
  const challenge = [
    'Practices and training do not always fit a traditional school day.',
    'Tournament travel can create academic gaps.',
    'Families need flexibility without losing accountability.',
  ];
  const credibility = [
    'Accredited K–12 online school',
    'Certified teachers',
    'Counselor support',
    'Parent progress visibility',
    'Flexible pacing and scheduling',
    'Free for many Indiana families via the Choice Scholarship',
  ];

  return (
    <section id="funding" style={{ background: '#F1ECE0' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: 'clamp(48px, 6vw, 76px) clamp(20px, 5vw, 72px)' }}>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(12, 1fr)',
          gap: 'clamp(24px, 3.5vw, 56px)',
          alignItems: 'start',
          marginBottom: 'clamp(28px, 3.5vw, 44px)',
        }} className="flex-intro">
          <div style={{ gridColumn: 'span 6' }}>
            <Eyebrow tone="navy">The INDILA approach</Eyebrow>
            <h2 className="display" style={{
              fontSize: 'clamp(28px, 3.4vw, 44px)',
              color: '#15355B',
              lineHeight: 1.05,
              letterSpacing: '-0.015em',
            }}>
              Flexibility Without Losing Structure
            </h2>
          </div>
          <div style={{ gridColumn: '7 / span 6' }}>
            <p style={{ fontSize: 'clamp(15px, 1.3vw, 17px)', color: '#1A2333', lineHeight: 1.7, fontWeight: 300, maxWidth: '560px' }}>
              Serious athletes often need more control over when and where
              school happens. INDILA gives families that flexibility while
              keeping the structure parents expect: certified teachers,
              counselor support, progress visibility, and an accredited
              K&ndash;12 program.
            </p>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '0',
          borderTop: '1.5px solid #15355B',
        }} className="flex-cols">

          <div className="flex-col" style={{
            paddingRight: 'clamp(20px, 2.5vw, 36px)',
            paddingTop: 'clamp(24px, 2.6vw, 36px)',
            paddingBottom: 'clamp(24px, 2.6vw, 36px)',
            borderRight: '1px solid #D9D1BD',
          }}>
            <h3 className="display" style={{
              fontSize: 'clamp(18px, 1.8vw, 22px)',
              fontWeight: 400,
              color: '#15355B',
              lineHeight: 1.25,
              marginBottom: '18px',
            }}>
              The challenge for serious athletes
            </h3>
            <ul style={{ listStyle: 'none' }}>
              {challenge.map(c => (
                <li key={c} style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                  padding: '10px 0',
                  fontSize: '14.5px',
                  color: '#1A2333',
                  lineHeight: 1.6,
                }}>
                  <span aria-hidden="true" style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '18px',
                    height: '18px',
                    background: '#15355B',
                    flexShrink: 0,
                    marginTop: '1px',
                  }}>
                    <Check size={11} color="#FAF7F0" strokeWidth={2.4} />
                  </span>
                  {c}
                </li>
              ))}
            </ul>
          </div>

          <div className="flex-col" style={{
            paddingLeft: 'clamp(20px, 2.5vw, 36px)',
            paddingTop: 'clamp(24px, 2.6vw, 36px)',
            paddingBottom: 'clamp(24px, 2.6vw, 36px)',
          }}>
            <h3 className="display" style={{
              fontSize: 'clamp(18px, 1.8vw, 22px)',
              fontWeight: 400,
              color: '#15355B',
              lineHeight: 1.25,
              marginBottom: '18px',
            }}>
              How INDILA keeps students on track
            </h3>
            <ul style={{ listStyle: 'none' }}>
              {credibility.map(item => (
                <li key={item} style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                  padding: '10px 0',
                  fontSize: '14.5px',
                  color: '#1A2333',
                  lineHeight: 1.55,
                }}>
                  <span aria-hidden="true" style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '18px',
                    height: '18px',
                    background: '#D4A029',
                    flexShrink: 0,
                    marginTop: '1px',
                  }}>
                    <Check size={11} color="#15355B" strokeWidth={2.6} />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <p style={{
          marginTop: 'clamp(20px, 2.5vw, 32px)',
          fontSize: '13px',
          color: '#5A6570',
          lineHeight: 1.7,
          maxWidth: '780px',
        }}>
          For many Indiana families, INDILA is free. The Indiana Choice
          Scholarship can cover tuition in full for eligible families.
          Admissions can walk you through whether your family qualifies
          and what the next steps look like.
        </p>

        <style>{`
          @media (max-width: 880px) {
            .flex-intro { display: block !important; }
            .flex-intro > div + div { margin-top: 18px; }
            .flex-cols { grid-template-columns: 1fr !important; }
            .flex-col { padding-left: 0 !important; padding-right: 0 !important; border-right: none !important; border-bottom: 1px solid #D9D1BD; }
            .flex-col:last-child { border-bottom: none; }
          }
        `}</style>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// (legacy SportManifest, Problem, and Solution components have been merged
// into FlexibilityWithStructure above)
// ═════════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 05 · HOW IT WORKS
// Five clean steps. Desktop: stepped grid with hairline rule. Mobile:
// vertical timeline with a left rail. Step ordering is explicit.
// ═════════════════════════════════════════════════════════════════════════════

function HowItWorks() {
  const steps = [
    { n: '01', t: 'Request Information',     b: 'Tell us about your student, schedule, and academic needs.' },
    { n: '02', t: 'Meet With Admissions',    b: 'Our team explains how INDILA works and helps determine fit.' },
    { n: '03', t: 'Build the Academic Plan', b: 'We identify courses, support needs, and schedule considerations.' },
    { n: '04', t: 'Begin With Support',      b: 'Students start coursework with teacher and counselor support.' },
    { n: '05', t: 'Keep Moving Forward',     b: 'Families stay informed while students make steady progress.' },
  ];

  return (
    <section id="how-it-works" style={{ background: '#0E2544', color: '#FAF7F0', position: 'relative' }}>
      <div aria-hidden="true" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(212,160,41,0.4), transparent)' }} />
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: 'clamp(48px, 6vw, 78px) clamp(20px, 5vw, 72px)' }}>

        <div style={{ maxWidth: '760px', marginBottom: 'clamp(28px, 3.5vw, 48px)' }}>
          <Eyebrow onDark>How It Works</Eyebrow>
          <h2 className="display" style={{
            fontSize: 'clamp(28px, 3.4vw, 44px)',
            color: '#FAF7F0',
            lineHeight: 1.05,
            letterSpacing: '-0.015em',
            marginBottom: '14px',
          }}>
            How It Works
          </h2>
          <p style={{ fontSize: '15px', color: 'rgba(250,247,240,0.72)', lineHeight: 1.7, fontWeight: 300, maxWidth: '620px' }}>
            A simple path from first inquiry to a school plan that works.
          </p>
        </div>

        <ol style={{ listStyle: 'none', display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 0, borderTop: '1px solid rgba(250,247,240,0.18)' }} className="hiw-grid">
          {steps.map((s, i) => (
            <li key={s.n} style={{
              padding: 'clamp(22px, 2.4vw, 30px) clamp(14px, 1.6vw, 22px) clamp(22px, 2.4vw, 32px) 0',
              paddingLeft: i === 0 ? 0 : 'clamp(14px, 1.6vw, 22px)',
              borderRight: i < steps.length - 1 ? '1px solid rgba(250,247,240,0.12)' : 'none',
              position: 'relative',
            }} className="hiw-step">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
                <span className="display" style={{ fontSize: '28px', fontWeight: 300, color: '#D4A029', lineHeight: 1, letterSpacing: '-0.02em' }}>
                  {s.n}
                </span>
                <span style={{ flex: 1, height: '1px', background: 'rgba(250,247,240,0.18)' }} />
              </div>
              <h3 className="display" style={{ fontSize: 'clamp(16px, 1.5vw, 19px)', fontWeight: 400, color: '#FAF7F0', lineHeight: 1.25, marginBottom: '8px' }}>
                {s.t}
              </h3>
              <p style={{ fontSize: '13px', color: 'rgba(250,247,240,0.65)', lineHeight: 1.65 }}>
                {s.b}
              </p>
            </li>
          ))}
        </ol>

        <div style={{ marginTop: 'clamp(28px, 3.5vw, 48px)' }}>
          <PrimaryCTA onDark href={FORM_ANCHOR_TOP}>Request Information</PrimaryCTA>
        </div>

        <style>{`
          @media (max-width: 1100px) {
            .hiw-grid { grid-template-columns: repeat(2, 1fr) !important; }
            .hiw-step { border-right: none !important; border-bottom: 1px solid rgba(250,247,240,0.12); padding: 24px 0 !important; }
            .hiw-step:nth-child(odd) { padding-right: clamp(14px, 2vw, 24px) !important; border-right: 1px solid rgba(250,247,240,0.12) !important; }
            .hiw-step:nth-child(even) { padding-left: clamp(14px, 2vw, 24px) !important; }
          }
          @media (max-width: 720px) {
            .hiw-grid { grid-template-columns: 1fr !important; }
            .hiw-step, .hiw-step:nth-child(odd), .hiw-step:nth-child(even) {
              border-right: none !important;
              padding: 22px 0 !important;
            }
            .hiw-step:last-child { border-bottom: none; }
          }
        `}</style>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 06 · ATHLETIC ASSOCIATION CREDIBILITY
// Speaks directly to clubs, associations, and training programs. Asymmetric
// layout with a strong CTA to "Discuss a Partnership" that scrolls to the
// final form so partnership inquiries route through the same lead pipeline.
// ═════════════════════════════════════════════════════════════════════════════

function AssociationCredibility() {
  const bullets = [
    'Helps families with demanding training and travel schedules',
    'Gives organizations a credible school option to share',
    'Keeps your program focused on athletics',
    'Gives families a direct path to admissions',
    'Supports flexibility without removing accountability',
  ];

  return (
    <section id="coaches" style={{ background: '#E5DEC9' }}>
      <div style={{
        maxWidth: '1280px',
        margin: '0 auto',
        padding: 'clamp(48px, 6vw, 76px) clamp(20px, 5vw, 72px)',
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(12, 1fr)',
          gap: 'clamp(28px, 3.5vw, 60px)',
          alignItems: 'start',
        }} className="assoc-grid">

          <div style={{ gridColumn: 'span 7' }}>
            <Eyebrow tone="navy">For coaches, clubs, and associations</Eyebrow>
            <h2 className="display" style={{
              fontSize: 'clamp(26px, 3vw, 38px)',
              color: '#15355B',
              lineHeight: 1.1,
              letterSpacing: '-0.015em',
              marginBottom: '20px',
            }}>
              A Credible Academic Resource for Athletic Organizations
            </h2>
            <p style={{ fontSize: '15px', color: '#1A2333', lineHeight: 1.7, fontWeight: 300, maxWidth: '600px', marginBottom: '14px' }}>
              Coaches, clubs, and athletic associations often meet families
              whose students need more flexibility than a traditional school
              schedule can provide. INDILA gives those families a credible
              academic option with structure, support, and flexibility.
            </p>
            <p style={{ fontSize: '14.5px', color: '#404C5E', lineHeight: 1.7, maxWidth: '600px' }}>
              INDILA supports the academic side, while your organization
              remains focused on training, competition, and athlete development.
            </p>
          </div>

          <div style={{ gridColumn: 'span 5' }}>
            <ul style={{ listStyle: 'none', borderTop: '1.5px solid #15355B' }}>
              {bullets.map(b => (
                <li key={b} style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '14px',
                  padding: '12px 0',
                  borderBottom: '1px solid #D9D1BD',
                  fontSize: '14px',
                  color: '#1A2333',
                  lineHeight: 1.55,
                }}>
                  <span aria-hidden="true" style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '20px',
                    height: '20px',
                    background: '#15355B',
                    flexShrink: 0,
                    marginTop: '2px',
                  }}>
                    <Check size={12} color="#D4A029" strokeWidth={2.5} />
                  </span>
                  {b}
                </li>
              ))}
            </ul>

            <div style={{ marginTop: '24px' }}>
              <PrimaryCTA href={FORM_ANCHOR_TOP}>Discuss a Partnership</PrimaryCTA>
            </div>
          </div>
        </div>

        <style>{`
          @media (max-width: 880px) {
            .assoc-grid { display: block !important; }
            .assoc-grid > div + div { margin-top: 36px; }
          }
        `}</style>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 08 · TESTIMONIAL PLACEHOLDERS
// IMPORTANT: These cards are PLACEHOLDERS. They are NOT real testimonials.
// Each card is labeled "PLACEHOLDER" prominently and the body text is
// obviously holding-copy. The whole section can be hidden with the
// SHOW_TESTIMONIAL_PLACEHOLDERS flag at the top of this file.
//
// TODO(testimonials): Replace these testimonial placeholders with approved
// real testimonials before publishing publicly. Until that happens, set
// SHOW_TESTIMONIAL_PLACEHOLDERS to false to hide this section in production.
// ═════════════════════════════════════════════════════════════════════════════

function TestimonialPlaceholders() {
  if (!SHOW_TESTIMONIAL_PLACEHOLDERS) return null;

  const cards = [
    {
      role: 'Parent testimonial placeholder',
      copy: 'Placeholder for future parent testimonial after launch.',
    },
    {
      role: 'Coach testimonial placeholder',
      copy: 'Placeholder for future coach or club director testimonial.',
    },
    {
      role: 'Association testimonial placeholder',
      copy: 'Placeholder for future athletic association testimonial.',
    },
  ];

  return (
    <section style={{ background: '#F1ECE0' }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: 'clamp(80px, 10vw, 132px) clamp(20px, 5vw, 72px)' }}>

        <div style={{ maxWidth: '720px', marginBottom: 'clamp(40px, 5vw, 64px)' }}>
          <Eyebrow tone="navy">Voices from our community</Eyebrow>
          <h2 className="display" style={{
            fontSize: 'clamp(28px, 3.2vw, 42px)',
            color: '#15355B',
            lineHeight: 1.08,
            letterSpacing: '-0.015em',
            marginBottom: '14px',
          }}>
            What Families and Partners Will Share.
          </h2>
          <p style={{ fontSize: '14.5px', color: '#5A6570', lineHeight: 1.7 }}>
            Placeholder section for future parent, student, coach, and
            association testimonials. Cards below are internal placeholders
            and are not real testimonials.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 'clamp(16px, 2vw, 24px)',
        }} className="testimonial-grid">
          {cards.map(c => (
            <article key={c.role} style={{
              position: 'relative',
              background: '#FFFFFF',
              padding: 'clamp(28px, 2.8vw, 36px)',
              borderTop: '3px dashed #9C2426',
              display: 'flex',
              flexDirection: 'column',
              minHeight: '220px',
            }}>
              <span style={{
                position: 'absolute',
                top: '-1px',
                right: '12px',
                background: '#9C2426',
                color: '#FAF7F0',
                padding: '4px 10px',
                fontSize: '9.5px',
                fontWeight: 700,
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
              }}>
                Placeholder
              </span>

              <div className="display" style={{
                fontSize: '36px',
                color: '#D4A029',
                opacity: 0.45,
                lineHeight: 1,
                marginBottom: '12px',
              }} aria-hidden="true">&ldquo;</div>

              <p style={{
                fontSize: '14.5px',
                color: '#5A6570',
                lineHeight: 1.7,
                marginBottom: '24px',
                flex: 1,
                fontStyle: 'italic',
              }}>
                {c.copy}
              </p>

              <div style={{ borderTop: '1px solid #D9D1BD', paddingTop: '16px' }}>
                <div style={{
                  fontSize: '10.5px',
                  fontWeight: 700,
                  letterSpacing: '0.20em',
                  textTransform: 'uppercase',
                  color: '#9C2426',
                }}>
                  {c.role}
                </div>
                <div style={{ fontSize: '12px', color: '#8D949C', marginTop: '4px' }}>
                  Replace before publishing publicly
                </div>
              </div>
            </article>
          ))}
        </div>

        <style>{`
          @media (max-width: 880px) {
            .testimonial-grid { grid-template-columns: 1fr !important; }
          }
        `}</style>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 08 · FAQ
// Seven questions with honest answers. Avoids NCAA, IHSAA, eligibility,
// recruitment, and scholarship guarantees. Hedges Indiana Choice Scholarship
// language ("may be eligible"), routes everything to admissions.
// ═════════════════════════════════════════════════════════════════════════════

type FaqItem = { q: string; a: string };

const FAQS: FaqItem[] = [
  {
    q: 'Is INDILA only for athletes?',
    a: 'No. INDILA serves students across Indiana, but this page is designed for families whose athletic schedules make traditional school difficult.',
  },
  {
    q: 'Is INDILA accredited?',
    a: 'Yes. INDILA is an accredited K–12 online school. Students receive a structured academic program with teacher and school support.',
  },
  {
    q: 'Can students work around practices, training, and tournaments?',
    a: 'Yes. INDILA gives students more flexibility in when and where they complete schoolwork while maintaining academic expectations and progress.',
  },
  {
    q: 'Do students have teachers and support?',
    a: 'Yes. Students are supported by teachers, counselors, administrators, and school staff. INDILA is not simply a library of online courses.',
  },
  {
    q: 'Can athletic associations or clubs refer families to INDILA?',
    a: 'Yes. Athletic associations, clubs, and training programs can share INDILA as an academic option for families who need more flexibility around training and competition.',
  },
  {
    q: 'Is INDILA free for Indiana families?',
    a: 'For many Indiana families, yes. The Indiana Choice Scholarship can cover INDILA tuition in full for eligible families. Admissions can walk you through whether your family qualifies and what the process looks like.',
  },
  {
    q: 'What is the best next step?',
    a: 'Fill out the form on this page or schedule a call with admissions. Our team will help you understand whether INDILA is the right fit for your student or athletic community.',
  },
];

function FAQ() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section id="faq" style={{ background: '#F1ECE0' }}>
      <div style={{ maxWidth: '1080px', margin: '0 auto', padding: 'clamp(48px, 6vw, 78px) clamp(20px, 5vw, 72px)' }}>

        <div style={{ marginBottom: 'clamp(28px, 4vw, 48px)', maxWidth: '720px' }}>
          <Eyebrow tone="navy">Questions parents and partners ask</Eyebrow>
          <h2 className="display" style={{
            fontSize: 'clamp(28px, 3.4vw, 44px)',
            color: '#15355B',
            lineHeight: 1.05,
            letterSpacing: '-0.015em',
          }}>
            Clear Answers. <span className="display-italic">No Fine Print.</span>
          </h2>
        </div>

        <div style={{ borderTop: '1px solid #D9D1BD' }}>
          {FAQS.map((item, i) => {
            const isOpen = open === i;
            return (
              <div key={i} style={{
                borderBottom: '1px solid #D9D1BD',
                borderLeft: isOpen ? '2px solid #D4A029' : '2px solid transparent',
                paddingLeft: isOpen ? '24px' : '0',
                transition: 'border-color 0.2s ease, padding-left 0.2s ease',
              }}>
                <button
                  onClick={() => setOpen(isOpen ? null : i)}
                  style={{
                    width: '100%',
                    background: 'transparent',
                    border: 'none',
                    padding: '17px 0',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '24px',
                    cursor: 'pointer',
                    textAlign: 'left',
                    color: '#15355B',
                    fontFamily: 'inherit',
                  }}
                  aria-expanded={isOpen}
                >
                  <span
                    className="display"
                    style={{
                      fontSize: 'clamp(16px, 1.5vw, 19px)',
                      fontWeight: 400,
                      letterSpacing: '-0.005em',
                      lineHeight: 1.3,
                    }}
                  >
                    {item.q}
                  </span>
                  <span style={{
                    flexShrink: 0,
                    width: '32px',
                    height: '32px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid ' + (isOpen ? '#D4A029' : '#D9D1BD'),
                    color: isOpen ? '#FAF7F0' : '#15355B',
                    background: isOpen ? '#D4A029' : 'transparent',
                    transition: 'all 0.2s ease',
                  }}>
                    {isOpen ? <Minus size={14} /> : <Plus size={14} />}
                  </span>
                </button>
                {isOpen && (
                  <div style={{ paddingBottom: '22px', paddingRight: 'clamp(20px, 4vw, 56px)' }}>
                    <p style={{ fontSize: '14.5px', color: '#5A6570', lineHeight: 1.75 }}>
                      {item.a}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: '28px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '20px' }}>
          <span style={{ fontSize: '14px', color: '#5A6570' }}>Still have questions?</span>
          <GhostLink href={FORM_ANCHOR_TOP}>Request Information</GhostLink>
        </div>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// SECTION 09 · COMPACT FINAL CTA
// Deepest navy. A short closing pitch with two CTAs and an email fallback.
// Replaces the prior bottom full form: the hero form is the single canonical
// lead surface, so the page stays decisive and the bottom does not repeat it.
// ═════════════════════════════════════════════════════════════════════════════

function CompactFinalCTA() {
  return (
    <section id="admissions" style={{ background: '#091528', color: '#FAF7F0', position: 'relative', overflow: 'hidden' }}>
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'radial-gradient(60% 80% at 50% 10%, rgba(212,160,41,0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />
      <div aria-hidden="true" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(212,160,41,0.35), transparent)' }} />

      <div style={{
        position: 'relative',
        maxWidth: '880px',
        margin: '0 auto',
        padding: 'clamp(56px, 7vw, 88px) clamp(20px, 5vw, 72px)',
        textAlign: 'center',
      }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
          <IndilaMark size={42} navy="#FAF7F0" />
        </div>

        <h2 className="display" style={{
          fontSize: 'clamp(26px, 3.2vw, 40px)',
          color: '#FAF7F0',
          lineHeight: 1.08,
          letterSpacing: '-0.015em',
          marginBottom: '18px',
        }}>
          Ready to Explore a More Flexible School Option?
        </h2>

        <p style={{
          fontSize: 'clamp(14.5px, 1.3vw, 16.5px)',
          color: 'rgba(250,247,240,0.78)',
          lineHeight: 1.7,
          fontWeight: 300,
          margin: '0 auto 28px',
          maxWidth: '620px',
        }}>
          Whether you are a parent, coach, club director, or athletic
          association leader, INDILA can help you understand whether a
          flexible accredited school option is the right fit.
        </p>

        <p style={{
          fontSize: 'clamp(13px, 1.1vw, 14.5px)',
          color: 'rgba(212,160,41,0.90)',
          lineHeight: 1.6,
          margin: '-12px auto 28px',
          maxWidth: '520px',
          fontWeight: 500,
        }}>
          For many Indiana families, INDILA is free through the Indiana Choice Scholarship.
        </p>

        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'center', gap: '14px', marginBottom: '20px' }}>
          <PrimaryCTA onDark href={FORM_ANCHOR_TOP}>Request Information</PrimaryCTA>
        </div>

        {/* TODO(admissions email): replace with the real admissions email when finalized. */}
        <p style={{ fontSize: '13px', color: 'rgba(250,247,240,0.55)' }}>
          Prefer email? <a href="mailto:admissions@indila.org" style={{ color: 'rgba(250,247,240,0.85)', textDecoration: 'none', borderBottom: '1px solid rgba(250,247,240,0.25)' }}>admissions@indila.org</a>
        </p>
      </div>
    </section>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// PAGE EXPORT
// ═════════════════════════════════════════════════════════════════════════════

export default function Athletics() {
  return (
    <>
      <Hero />
      <AudienceSplit />
      <FlexibilityWithStructure />
      <AssociationCredibility />
      <HowItWorks />
      <TestimonialPlaceholders />
      <FAQ />
      <CompactFinalCTA />
    </>
  );
}
