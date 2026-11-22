"""Read the active Windows Mobile Hotspot configuration for local QR sharing."""

import json
import subprocess
import sys
from dataclasses import dataclass
from ipaddress import IPv4Address


@dataclass(frozen=True)
class Hotspot:
    address: str
    ssid: str
    password: str


class HotspotAddressUnavailable(OSError):
    """The hotspot is on, but its Wi-Fi adapter has no controller address yet."""


QUERY = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$network = [Windows.Networking.Connectivity.NetworkInformation, Windows, ContentType=WindowsRuntime]
$manager = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows, ContentType=WindowsRuntime]
$profiles = @($network::GetInternetConnectionProfile()) + @($network::GetConnectionProfiles())
$checked = $false
foreach ($profile in $profiles) {
    if ($null -eq $profile) { continue }
    try {
        $tether = $manager::CreateFromConnectionProfile($profile)
        $state = $tether.TetheringOperationalState
        $checked = $true
    } catch { continue }
    if ($state -ne 'On') { continue }

    $config = $tether.GetCurrentAccessPointConfiguration()
    $direct = [Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
        Where-Object { $_.OperationalStatus -eq 'Up' -and $_.Description -like '*Wi-Fi Direct*' }
    $addresses = @($direct | ForEach-Object { $_.GetIPProperties().UnicastAddresses } |
        Where-Object { $_.Address.AddressFamily -eq 'InterNetwork' -and
                       $_.DuplicateAddressDetectionState -eq 'Preferred' } |
        ForEach-Object { $_.Address.ToString() })
    $address = $addresses | Where-Object { $_ -eq '192.168.137.1' } | Select-Object -First 1
    if (-not $address) {
        $address = $addresses | Where-Object { $_ -notlike '169.254.*' } | Select-Object -First 1
    }
    @{state='on'; address=$address; ssid=$config.Ssid; password=$config.Passphrase} |
        ConvertTo-Json -Compress
    exit
}
if ($profiles.Count -le 1 -or $checked) { '{"state":"off"}' }
else { '{"state":"unavailable"}' }
"""


def read_hotspot() -> Hotspot | None:
    if sys.platform != "win32":
        raise OSError("Windows Mobile Hotspot is available on Windows.")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", QUERY],
        capture_output=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=True,
    )
    data = json.loads(result.stdout.decode("utf-8-sig"))
    if data["state"] == "off":
        return None
    if data["state"] != "on":
        raise OSError("Windows did not report a ready Wi-Fi hotspot.")
    address = IPv4Address(data["address"]) if data.get("address") else None
    if address is None or address.is_link_local or address.is_unspecified:
        raise HotspotAddressUnavailable()
    if not data.get("ssid") or not data.get("password"):
        raise OSError("Windows did not report hotspot credentials.")
    return Hotspot(str(address), data["ssid"], data["password"])
