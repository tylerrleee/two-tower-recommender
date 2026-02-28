# Allow 10-50 active connections - pooled connection
minPoolSize : int   = 10
maxPoolSize : int   = 50
maxIdleTimeMS : int = 45000  

# Failover & Timeouts
serveSelectionTimeoutMS : int = 5000 # if Atlas cluster is unavailable for 5s
connectTimeoutMS : int        = 10000 # Timeout connection for 10s
socketTimeoutMS  : int        = 20000

