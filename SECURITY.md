# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | ✅ Yes             |
| 0.1.x   | ❌ No              |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in NeuroIndex, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please email security concerns to: **umeshkumar667@gmail.com**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Investigation**: We will investigate and validate the report
3. **Updates**: We will keep you informed of our progress
4. **Resolution**: We aim to resolve critical issues within 7 days
5. **Credit**: We will credit you in the security advisory (unless you prefer anonymity)

### Scope

In scope:
- SQL injection vulnerabilities
- Path traversal vulnerabilities
- Data corruption issues
- Denial of service vulnerabilities
- Authentication/authorization bypasses

Out of scope:
- Issues in dependencies (report to those projects)
- Social engineering attacks
- Physical security

## Security Considerations

### Data Storage

NeuroIndex stores data locally using:
- **SQLite**: For document metadata and vectors
- **Pickle files**: For graph structure and FAISS mappings

**Recommendations**:
- Store data in secure, access-controlled directories
- Use filesystem encryption for sensitive data
- Regularly backup your data directory

### Input Validation

NeuroIndex validates:
- Vector dimensions
- Text content (non-empty)
- Numeric values (no NaN/Inf)

**Note**: NeuroIndex does not sanitize text content for downstream use. If you're displaying text in web applications, apply appropriate escaping.

### Pickle Security

NeuroIndex uses pickle for some persistence. Only load NeuroIndex data from trusted sources, as malicious pickle files can execute arbitrary code.

### Thread Safety

- Read operations are thread-safe
- Concurrent write operations require external synchronization

## Best Practices

1. **Keep Updated**: Always use the latest version
2. **Secure Storage**: Protect your data directory
3. **Trust Boundaries**: Only load indices from trusted sources
4. **Input Validation**: Validate user inputs before storing
5. **Access Control**: Restrict access to the storage directory

