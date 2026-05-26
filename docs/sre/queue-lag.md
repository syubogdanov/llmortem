# Queue Lag Incident

## Symptoms

- Queue lag is growing.
- Workers are running but messages are processed slowly.
- Users report delayed notifications or delayed background tasks.

## Checks

1. Check current lag in monitoring.
2. Check worker logs for repeated errors.
3. Check whether consumers are running.
4. Check database and external API latency.
5. Check whether a recent release changed message format.

## Mitigation

1. Pause non-critical producers if lag grows too quickly.
2. Restart unhealthy workers if logs show stuck consumers.
3. Scale worker replicas if CPU and database capacity allow it.
4. Roll back the latest release if lag started immediately after deployment.

## Escalation

Escalate to the service owner if lag keeps growing for more than 15 minutes after mitigation.