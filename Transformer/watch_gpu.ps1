Get-Content gpu_profile.log -Wait | ForEach-Object {
    "$(Get-Date -Format 'HH:mm:ss') $_"
}
