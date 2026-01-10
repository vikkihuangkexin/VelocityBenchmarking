
chown -R user1:user1 /data/simdata 2>/dev/null || true
exec gosu user1 "$@"