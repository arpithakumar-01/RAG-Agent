# Engineering Standards

## Code Review
Every pull request requires at least 1 approving review before merging. Pull requests that change infrastructure or security configuration require 2 approvals, including one from the platform team. Authors may not approve their own changes.

## Testing
New services must maintain at least 80 percent unit test coverage. Pull requests that reduce coverage below this threshold are blocked by the CI pipeline.

## Deployments
All deployments go through the CI/CD pipeline; manual production deploys are not allowed. There is a deploy freeze every Friday after 3:00 PM local time and during company holidays. Emergency fixes during a freeze need approval from the engineering director.

## Secrets Management
Secrets such as API keys and database passwords must be stored in the secret manager and never committed to source control. Secrets must be rotated every 90 days. A leaked secret must be revoked within 1 hour of discovery.
