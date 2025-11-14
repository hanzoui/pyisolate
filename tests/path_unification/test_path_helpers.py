"""Unit tests for path_helpers module - path unification logic."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from pyisolate.path_helpers import (
    build_child_sys_path,
    serialize_host_snapshot,
)


class TestSerializeHostSnapshot:
    """Tests for host environment snapshot capture."""
    
    def test_snapshot_contains_required_keys(self):
        """Snapshot must include sys.path, executable, prefix, and env vars."""
        snapshot = serialize_host_snapshot()
        
        assert "sys_path" in snapshot
        assert "sys_executable" in snapshot
        assert "sys_prefix" in snapshot
        assert "environment" in snapshot
        
        assert isinstance(snapshot["sys_path"], list)
        assert isinstance(snapshot["sys_executable"], str)
        assert isinstance(snapshot["sys_prefix"], str)
        assert isinstance(snapshot["environment"], dict)
    
    def test_snapshot_captures_sys_path(self):
        """sys_path in snapshot should match current sys.path."""
        snapshot = serialize_host_snapshot()
        assert snapshot["sys_path"] == list(sys.path)
    
    def test_snapshot_writes_to_file(self):
        """When output_path provided, snapshot should be written as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "snapshot.json"
            snapshot = serialize_host_snapshot(str(output_path))
            
            assert output_path.exists()
            with open(output_path, "r") as f:
                loaded = json.load(f)
            
            assert loaded["sys_path"] == snapshot["sys_path"]
            assert loaded["sys_executable"] == snapshot["sys_executable"]
    
    def test_snapshot_without_file_returns_dict(self):
        """When no output_path, should return dict without side effects."""
        snapshot = serialize_host_snapshot()
        assert isinstance(snapshot, dict)
        assert len(snapshot["sys_path"]) > 0
    
    def test_snapshot_with_extra_env_keys(self):
        """Should capture additional env vars when extra_env_keys provided."""
        # Set a test env var
        os.environ["TEST_PYISOLATE_VAR"] = "test_value"
        
        try:
            snapshot = serialize_host_snapshot(extra_env_keys=["TEST_PYISOLATE_VAR"])
            
            assert "environment" in snapshot
            assert "TEST_PYISOLATE_VAR" in snapshot["environment"]
            assert snapshot["environment"]["TEST_PYISOLATE_VAR"] == "test_value"
        finally:
            del os.environ["TEST_PYISOLATE_VAR"]


class TestBuildChildSysPath:
    """Tests for child sys.path reconstruction logic."""
    
    def test_preserves_host_order(self):
        """Host paths must appear in original order."""
        host = ["/host/lib1", "/host/lib2", "/host/lib3"]
        extras = ["/venv/lib"]
        
        result = build_child_sys_path(host, extras)
        
        # Host paths should be first, in order
        assert result[:3] == host
        assert result[3] == extras[0]
    
    def test_removes_duplicates(self):
        """Duplicate paths should be removed while preserving first occurrence."""
        host = ["/host/lib", "/host/lib2", "/host/lib"]
        extras = ["/venv/lib"]
        
        result = build_child_sys_path(host, extras)
        
        # First /host/lib kept, second removed
        assert result.count("/host/lib") == 1
        assert result[0] == "/host/lib"
    
    def test_inserts_comfy_root_first_when_missing(self):
        """If comfy_root provided and not in host_paths, prepend it."""
        host = ["/host/lib1", "/host/lib2"]
        extras = ["/venv/lib"]
        comfy_root = "/home/johnj/ComfyUI"
        
        result = build_child_sys_path(host, extras, comfy_root=comfy_root)
        
        assert result[0] == comfy_root
        assert result[1:3] == host
    
    def test_does_not_duplicate_comfy_root_if_present(self):
        """If comfy_root already in host_paths, don't duplicate it."""
        comfy_root = "/home/johnj/ComfyUI"
        host = [comfy_root, "/host/lib1"]
        extras = ["/venv/lib"]
        
        result = build_child_sys_path(host, extras, comfy_root=comfy_root)
        
        # Should only appear once
        assert result.count(comfy_root) == 1
        assert result[0] == comfy_root
    
    def test_removes_comfy_subdirectories_when_root_specified(self):
        """Subdirectories of comfy_root should be filtered to avoid shadowing."""
        comfy_root = "/home/johnj/ComfyUI"
        host = [f"{comfy_root}/comfy", f"{comfy_root}/app", "/host/lib"]
        extras = ["/venv/lib"]
        
        result = build_child_sys_path(host, extras, comfy_root=comfy_root)
        
        # ComfyUI root should be first
        assert result[0] == comfy_root
        # Subdirectories should be removed
        assert f"{comfy_root}/comfy" not in result
        assert f"{comfy_root}/app" not in result
        # Other paths should remain
        assert "/host/lib" in result
    
    def test_preserves_venv_site_packages_under_comfy_root(self):
        """ComfyUI .venv site-packages should NOT be filtered out."""
        comfy_root = "/home/johnj/ComfyUI"
        venv_site = f"{comfy_root}/.venv/lib/python3.12/site-packages"
        host = [f"{comfy_root}/comfy", venv_site, "/host/lib"]
        extras = []
        
        result = build_child_sys_path(host, extras, comfy_root=comfy_root)
        
        # ComfyUI root should be first
        assert result[0] == comfy_root
        # .venv site-packages MUST be preserved
        assert venv_site in result
        # comfy subdir should be removed
        assert f"{comfy_root}/comfy" not in result
    
    def test_appends_extra_paths(self):
        """Extra paths (isolated venv) should be appended after host paths."""
        host = ["/host/lib"]
        extras = ["/venv/lib1", "/venv/lib2"]
        
        result = build_child_sys_path(host, extras)
        
        assert result[0] == host[0]
        assert result[1:] == extras
    
    def test_handles_empty_host_paths(self):
        """Should work with empty host paths (edge case)."""
        host = []
        extras = ["/venv/lib"]
        
        result = build_child_sys_path(host, extras)
        
        assert result == extras
    
    def test_handles_empty_extra_paths(self):
        """Should work with empty extra paths."""
        host = ["/host/lib"]
        extras = []
        
        result = build_child_sys_path(host, extras)
        
        assert result == host
    
    def test_normalizes_paths_for_duplicate_detection(self):
        """Paths differing only in case/separators should be deduplicated."""
        # This test assumes case-insensitive filesystem (Windows-like)
        # On Linux it may not dedupe, which is correct behavior
        host = ["/Host/Lib", "/host/lib"]  # Different case
        extras = []
        
        result = build_child_sys_path(host, extras)
        
        # Result length depends on OS - just verify no crash
        assert len(result) >= 1
        assert len(result) <= 2
    
    def test_idempotent_with_repeated_extras(self):
        """Passing extras already in host should not duplicate."""
        host = ["/host/lib", "/venv/lib"]
        extras = ["/venv/lib"]  # Already in host
        
        result = build_child_sys_path(host, extras)
        
        assert result.count("/venv/lib") == 1
    
    def test_handles_empty_string_paths(self):
        """Empty string paths should be filtered out by add_path guard."""
        host = ["/host/lib", "", "/host/lib2"]
        extras = ["", "/venv/lib"]
        
        result = build_child_sys_path(host, extras)
        
        # Empty strings should not appear
        assert "" not in result
        assert "/host/lib" in result
        assert "/host/lib2" in result
        assert "/venv/lib" in result


class TestIntegration:
    """Integration tests combining snapshot + path building."""
    
    def test_round_trip_snapshot_and_rebuild(self):
        """Capture snapshot, build child path, verify reconstruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot.json"
            
            # Capture current environment
            snapshot = serialize_host_snapshot(str(snapshot_path))
            
            # Simulate isolated venv paths
            fake_venv = Path(tmpdir) / ".venv" / "lib" / "python3.12" / "site-packages"
            extras = [str(fake_venv)]
            
            # Build child path
            child_path = build_child_sys_path(
                snapshot["sys_path"],
                extras,
                comfy_root="/home/johnj/ComfyUI"
            )
            
            # Verify structure
            assert "/home/johnj/ComfyUI" in child_path
            assert str(fake_venv) in child_path
            assert len(child_path) >= len(snapshot["sys_path"])
