Here's a concise summary of your dual git configuration setup:

## Dual Git Configuration Summary

### Configuration Structure

Three configuration files work together to automatically switch git identities:

1. **`~/.gitconfig`** - Main configuration with conditional includes
2. **`~/.gitconfig-work`** - Work-specific settings
3. **`~/.gitconfig-personal`** - Personal-specific settings

### How It Works

Git automatically applies the correct identity based on the repository's directory location using `includeIf` directives:

```gitconfig
[includeIf "gitdir:/Users/yosii/work/git_cp/"]
    path = ~/.gitconfig-work

[includeIf "gitdir:/Users/yosii/work/git/"]
    path = ~/.gitconfig-personal
```

### Identity Mapping

| Location | Name | Email | Account |
|----------|------|-------|---------|
| `/Users/yosii/work/git/` | yosi izaq | izaqyos@gmail.com | Personal (GitHub) |
| `/Users/yosii/work/git_cp/` | yosii | yosii@checkpoint.com | Work (Checkpoint) |

### Key Points

- **Automatic switching**: No manual configuration needed per repo
- **Directory-based**: Identity is determined by repository location
- **Works inside git repos only**: The conditional includes activate when you're inside an initialized git repository
- **Default identity**: Personal account is set as the default fallback
- **No SSH keys**: Configuration uses HTTPS authentication

### Verification Command

To check which identity is active in any repository:
```bash
git config user.name
git config user.email
```

### Maintenance

To modify settings, edit the files directly or use:
```bash
git config --global --edit          # Edit main config
vim ~/.gitconfig-work               # Edit work config
vim ~/.gitconfig-personal           # Edit personal config
```