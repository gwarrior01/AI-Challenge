use std::{collections::HashMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct User {
    pub id: String,
    pub name: String,
    pub email: String,
    pub plan: String,
    pub status: String,
    #[serde(default)]
    pub locale: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct TicketEvent {
    pub at: String,
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Ticket {
    pub id: String,
    pub user_id: String,
    pub subject: String,
    pub category: String,
    pub status: String,
    pub priority: String,
    pub created_at: String,
    #[serde(default)]
    pub environment: HashMap<String, String>,
    #[serde(default)]
    pub events: Vec<TicketEvent>,
}

/// One line of the ticket list: enough to pick a ticket, without its history.
#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub struct TicketSummary {
    pub id: String,
    pub user_id: String,
    pub user_name: String,
    pub subject: String,
    pub status: String,
    pub priority: String,
    pub created_at: String,
}

#[derive(Clone)]
pub struct Store {
    users: HashMap<String, User>,
    tickets: HashMap<String, Ticket>,
}

impl Store {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let users: Vec<User> = read_json(&dir.join("users.json"))?;
        let tickets: Vec<Ticket> = read_json(&dir.join("tickets.json"))?;

        Self::from_records(users, tickets)
    }

    fn from_records(users: Vec<User>, tickets: Vec<Ticket>) -> Result<Self> {
        let mut users_by_id = HashMap::new();
        for user in users {
            let id = user.id.clone();
            if users_by_id.insert(id.clone(), user).is_some() {
                bail!("повторяющийся идентификатор пользователя {id}");
            }
        }

        let mut tickets_by_id = HashMap::new();
        for ticket in tickets {
            let id = ticket.id.clone();
            if tickets_by_id.insert(id.clone(), ticket).is_some() {
                bail!("повторяющийся идентификатор тикета {id}");
            }
        }

        for ticket in tickets_by_id.values() {
            if !users_by_id.contains_key(&ticket.user_id) {
                bail!(
                    "тикет {} ссылается на неизвестного пользователя {}",
                    ticket.id,
                    ticket.user_id
                );
            }
        }
        Ok(Self {
            users: users_by_id,
            tickets: tickets_by_id,
        })
    }

    pub fn user(&self, id: &str) -> Option<User> {
        self.users.get(id).cloned()
    }

    pub fn ticket(&self, id: &str) -> Option<Ticket> {
        self.tickets.get(id).cloned()
    }

    /// All tickets (or only those with `status`), newest first.
    pub fn tickets(&self, status: Option<&str>) -> Vec<TicketSummary> {
        let mut tickets: Vec<_> = self
            .tickets
            .values()
            .filter(|ticket| status.is_none_or(|s| ticket.status.eq_ignore_ascii_case(s)))
            .map(|ticket| TicketSummary {
                id: ticket.id.clone(),
                user_id: ticket.user_id.clone(),
                user_name: self
                    .users
                    .get(&ticket.user_id)
                    .map(|u| u.name.clone())
                    .unwrap_or_default(),
                subject: ticket.subject.clone(),
                status: ticket.status.clone(),
                priority: ticket.priority.clone(),
                created_at: ticket.created_at.clone(),
            })
            .collect();
        tickets.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        tickets
    }

    pub fn user_tickets(&self, user_id: &str) -> Vec<Ticket> {
        let mut tickets: Vec<_> = self
            .tickets
            .values()
            .filter(|ticket| ticket.user_id == user_id)
            .cloned()
            .collect();
        tickets.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        tickets
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes =
        fs::read(path).with_context(|| format!("не удалось прочитать {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("неверный JSON в {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(id: &str) -> User {
        User {
            id: id.into(),
            name: "Тестовый пользователь".into(),
            email: "user@example.com".into(),
            plan: "Free".into(),
            status: "active".into(),
            locale: None,
            created_at: None,
            notes: Vec::new(),
        }
    }

    fn ticket(id: &str, user_id: &str, created_at: &str) -> Ticket {
        Ticket {
            id: id.into(),
            user_id: user_id.into(),
            subject: "Тестовый тикет".into(),
            category: "test".into(),
            status: "open".into(),
            priority: "normal".into(),
            created_at: created_at.into(),
            environment: HashMap::new(),
            events: Vec::new(),
        }
    }

    #[test]
    fn demo_data_is_consistent() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../support/data");
        let store = Store::load(dir).unwrap();
        let ticket = store.ticket("TCK-1001").unwrap();
        assert_eq!(ticket.user_id, "USR-001");
        assert_eq!(store.user(&ticket.user_id).unwrap().plan, "Pro");
        assert_eq!(store.user_tickets("USR-001").len(), 2);
        assert!(store.ticket("TCK-9999").is_none());
        // USR-002 has paid for Pro, but the payment is on hold.
        let ivan = store.user("USR-002").unwrap();
        assert_eq!(
            (ivan.plan.as_str(), ivan.status.as_str()),
            ("Pro", "payment_hold")
        );
    }

    #[test]
    fn ticket_list_filters_by_status() {
        let store = Store::from_records(
            vec![user("USR-1")],
            vec![
                ticket("TCK-OLD", "USR-1", "2026-01-01T00:00:00Z"),
                ticket("TCK-NEW", "USR-1", "2026-02-01T00:00:00Z"),
            ],
        )
        .unwrap();
        let all: Vec<_> = store.tickets(None).into_iter().map(|t| t.id).collect();
        assert_eq!(all, ["TCK-NEW", "TCK-OLD"]);
        assert_eq!(store.tickets(Some("OPEN")).len(), 2);
        assert!(store.tickets(Some("resolved")).is_empty());
        assert_eq!(store.tickets(None)[0].user_name, "Тестовый пользователь");
    }

    #[test]
    fn tickets_are_sorted_newest_first() {
        let store = Store::from_records(
            vec![user("USR-1")],
            vec![
                ticket("TCK-OLD", "USR-1", "2026-01-01T00:00:00Z"),
                ticket("TCK-NEW", "USR-1", "2026-02-01T00:00:00Z"),
            ],
        )
        .unwrap();

        let ids: Vec<_> = store
            .user_tickets("USR-1")
            .into_iter()
            .map(|ticket| ticket.id)
            .collect();
        assert_eq!(ids, ["TCK-NEW", "TCK-OLD"]);
    }

    #[test]
    fn rejects_ticket_with_unknown_user() {
        let error = Store::from_records(
            vec![user("USR-1")],
            vec![ticket("TCK-1", "USR-MISSING", "2026-01-01T00:00:00Z")],
        )
        .err()
        .unwrap();

        assert!(error.to_string().contains("USR-MISSING"));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let user_error = Store::from_records(vec![user("USR-1"), user("USR-1")], Vec::new())
            .err()
            .unwrap();
        assert!(user_error.to_string().contains("USR-1"));

        let ticket_error = Store::from_records(
            vec![user("USR-1")],
            vec![
                ticket("TCK-1", "USR-1", "2026-01-01T00:00:00Z"),
                ticket("TCK-1", "USR-1", "2026-01-02T00:00:00Z"),
            ],
        )
        .err()
        .unwrap();
        assert!(ticket_error.to_string().contains("TCK-1"));
    }
}
