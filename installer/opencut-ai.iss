; ===========================================================================
; OpenCut AI - Windows installer (Inno Setup 6)
;
; Produces OpenCutAI-Setup.exe: a small, per-user (no admin) bootstrapper that
; installs the launcher scripts, adds Start Menu / optional desktop shortcuts,
; and can run the setup immediately. The launcher (install.cmd -> install.ps1)
; clones the repo and brings up the Docker stack - the same flow as running the
; scripts by hand, but delivered as a double-click .exe so users never hit the
; PowerShell "running scripts is disabled" execution-policy error.
;
; Build locally:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\opencut-ai.iss
; or via CI (.github/workflows/windows-installer.yml). The version comes from the
; OPENCUT_VERSION environment variable (falls back to 0.1.0).
; ===========================================================================

#define MyAppName "OpenCut AI"
#define MyAppPublisher "OpenCut AI"
#define MyAppURL "https://github.com/Ekaanth/OpenCut-AI"

; Version from the environment (set by CI from the git tag); fallback for local builds.
#define MyAppVersion GetEnv("OPENCUT_VERSION")
#if MyAppVersion == ""
  #define MyAppVersion "0.1.0"
#endif

[Setup]
; A fixed AppId keeps upgrades/uninstall stable across versions - do not change it.
AppId={{8F2C1A6E-3B4D-4E5F-9A1B-7C2D3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
; Per-user install: no administrator rights required.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\OpenCut AI
DefaultGroupName=OpenCut AI
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir=Output
OutputBaseFilename=OpenCutAI-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\scripts\install.cmd

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; The launcher scripts. install.cmd is the entry point; install.ps1 does the work.
; run-native.py is bundled too, though the cloned repo also carries its own copy.
Source: "..\scripts\install.cmd"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\scripts\install.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\scripts\run-native.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
; Shortcuts launch install.cmd from the user's Documents folder, so the repo is
; cloned to %USERPROFILE%\Documents\OpenCut-AI and the Docker stack runs there.
Name: "{group}\OpenCut AI"; Filename: "{app}\scripts\install.cmd"; WorkingDir: "{userdocs}"; Comment: "Set up / launch OpenCut AI (requires Docker Desktop)"
Name: "{group}\OpenCut AI (AMD ROCm)"; Filename: "{app}\scripts\install.cmd"; Parameters: "-Rocm"; WorkingDir: "{userdocs}"; Comment: "Launch with AMD ROCm GPU acceleration"
Name: "{group}\OpenCut AI (NVIDIA)"; Filename: "{app}\scripts\install.cmd"; Parameters: "-Nvidia"; WorkingDir: "{userdocs}"; Comment: "Launch with NVIDIA GPU acceleration"
Name: "{group}\Uninstall OpenCut AI"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OpenCut AI"; Filename: "{app}\scripts\install.cmd"; WorkingDir: "{userdocs}"; Tasks: desktopicon

[Run]
; Offer to run the bootstrap right after install (unchecked-by-default would need
; 'unchecked'; we leave it checked so first-run "just works"). WorkingDir matches
; the shortcuts so the clone lands in the same place.
Filename: "{app}\scripts\install.cmd"; WorkingDir: "{userdocs}"; Description: "Set up and launch OpenCut AI now (needs Docker Desktop running)"; Flags: postinstall shellexec skipifsilent nowait
