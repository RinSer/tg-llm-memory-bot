package store

import "testing"

func TestGetCheckpointMissingIsZeroValue(t *testing.T) {
	s := newTestStore(t)
	c, err := s.GetCheckpoint(999)
	if err != nil {
		t.Fatalf("GetCheckpoint: %v", err)
	}
	if c.SessionID != 999 || c.LastSummarizedMessageID != 0 || c.LastError != "" {
		t.Fatalf("expected zero-value checkpoint, got %+v", c)
	}
}

func TestSaveSummarizationBatchAdvancesCheckpoint(t *testing.T) {
	s := newTestStore(t)

	if err := s.SaveSummarizationBatch(1, 10, []Fact{{Content: "f1", Embedding: []float32{1}}}, 42); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}

	c, err := s.GetCheckpoint(10)
	if err != nil {
		t.Fatalf("GetCheckpoint: %v", err)
	}
	if c.LastSummarizedMessageID != 42 {
		t.Fatalf("expected checkpoint advanced to 42, got %d", c.LastSummarizedMessageID)
	}
	if c.UserID != 1 {
		t.Fatalf("expected user id 1, got %d", c.UserID)
	}

	// A second batch advances further.
	if err := s.SaveSummarizationBatch(1, 10, []Fact{{Content: "f2", Embedding: []float32{2}}}, 99); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}
	c, _ = s.GetCheckpoint(10)
	if c.LastSummarizedMessageID != 99 {
		t.Fatalf("expected checkpoint advanced to 99, got %d", c.LastSummarizedMessageID)
	}
}

func TestRecordSummarizationErrorLeavesCheckpointUntouched(t *testing.T) {
	s := newTestStore(t)

	if err := s.SaveSummarizationBatch(1, 10, []Fact{{Content: "f1", Embedding: []float32{1}}}, 42); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}

	if err := s.RecordSummarizationError(1, 10, "boom"); err != nil {
		t.Fatalf("RecordSummarizationError: %v", err)
	}

	c, _ := s.GetCheckpoint(10)
	if c.LastSummarizedMessageID != 42 {
		t.Fatalf("expected message id to stay at 42 after error, got %d", c.LastSummarizedMessageID)
	}
	if c.LastError != "boom" || c.RetryCount != 1 {
		t.Fatalf("expected error 'boom' and retry_count 1, got %q / %d", c.LastError, c.RetryCount)
	}

	// A second error increments retry_count.
	if err := s.RecordSummarizationError(1, 10, "boom2"); err != nil {
		t.Fatalf("RecordSummarizationError: %v", err)
	}
	c, _ = s.GetCheckpoint(10)
	if c.RetryCount != 2 || c.LastError != "boom2" {
		t.Fatalf("expected retry_count 2 and error 'boom2', got %d / %q", c.RetryCount, c.LastError)
	}

	// A successful batch clears the error and retry state.
	if err := s.SaveSummarizationBatch(1, 10, nil, 50); err != nil {
		t.Fatalf("SaveSummarizationBatch: %v", err)
	}
	c, _ = s.GetCheckpoint(10)
	if c.LastError != "" || c.RetryCount != 0 || c.LastSummarizedMessageID != 50 {
		t.Fatalf("expected cleared error/retry and advanced checkpoint, got %+v", c)
	}
}

func TestRecordSummarizationErrorBeforeAnyBatch(t *testing.T) {
	s := newTestStore(t)
	if err := s.RecordSummarizationError(1, 10, "boom"); err != nil {
		t.Fatalf("RecordSummarizationError: %v", err)
	}
	c, _ := s.GetCheckpoint(10)
	if c.LastSummarizedMessageID != 0 || c.LastError != "boom" || c.RetryCount != 1 {
		t.Fatalf("unexpected checkpoint: %+v", c)
	}
}
