package sqlx

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"

	"github.com/jackc/pgconn"
	"github.com/jackc/pgx/v4"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/jmoiron/sqlx/reflectx"
)

// PgxQueryer is an interface used by Get and Select
type PgxQueryer interface {
	Query(ctx context.Context, query string, args ...interface{}) (pgx.Rows, error)
	Queryx(ctx context.Context, query string, args ...interface{}) (*PgxRows, error)
	QueryRowx(ctx context.Context, query string, args ...interface{}) *PgxRow
}

// PgxExecer is an interface used by MustExec and LoadFile
type PgxExecer interface {
	Exec(ctx context.Context, query string, args ...interface{}) (pgconn.CommandTag, error)
}

// PgxExt is a union interface which can bind, query, and exec, used by
// NamedPgxQuery and NamedPgxExec.
type PgxExt interface {
	binder
	PgxQueryer
	PgxExecer
}

// PgxRow is a reimplementation of pgx.Row in order to gain access to the underlying
// sql.Rows.Columns() data, necessary for StructScan.
type PgxRow struct {
	err    error
	unsafe bool
	rows   pgx.Rows
	Mapper *reflectx.Mapper
}

// Scan is a fixed implementation of sql.Row.Scan, which does not discard the
// underlying error from the internal rows object if it exists.
func (r *PgxRow) Scan(dest ...interface{}) error {
	if r.err != nil {
		return r.err
	}

	// TODO(bradfitz): for now we need to defensively clone all
	// []byte that the driver returned (not permitting
	// *RawBytes in Rows.Scan), since we're about to close
	// the Rows in our defer, when we return from this function.
	// the contract with the driver.Next(...) interface is that it
	// can return slices into read-only temporary memory that's
	// only valid until the next Scan/Close.  But the TODO is that
	// for a lot of drivers, this copy will be unnecessary.  We
	// should provide an optional interface for drivers to
	// implement to say, "don't worry, the []bytes that I return
	// from Next will not be modified again." (for instance, if
	// they were obtained from the network anyway) But for now we
	// don't care.
	defer r.rows.Close()
	for _, dp := range dest {
		if _, ok := dp.(*sql.RawBytes); ok {
			return errors.New("sql: RawBytes isn't allowed on Row.Scan")
		}
	}

	if !r.rows.Next() {
		if err := r.rows.Err(); err != nil {
			return err
		}
		return sql.ErrNoRows
	}
	err := r.rows.Scan(dest...)
	if err != nil {
		return err
	}
	return nil
}

// Columns returns the underlying sql.Rows.Columns(), or the deferred error usually
// returned by Row.Scan()
func (r *PgxRow) Columns() ([]string, error) {
	if r.err != nil {
		return []string{}, r.err
	}

	// From github.com/jackc/pgx/stdlib/sql.go method Columns
	fieldDescriptions := r.rows.FieldDescriptions()
	names := make([]string, 0, len(fieldDescriptions))
	for _, fd := range fieldDescriptions {
		names = append(names, string(fd.Name))
	}
	return names, nil
}

// Err returns the error encountered while scanning.
func (r *PgxRow) Err() error {
	return r.err
}

// PgxPool is a wrapper around pgxpool.Pool which keeps track of the driverName upon Open,
// used mostly to automatically bind named queries using the right bindvars.
type PgxPool struct {
	*pgxpool.Pool
	driverName string
	unsafe     bool
	Mapper     *reflectx.Mapper
}

// NewPgxPool wraps a pgx connection pool
func NewPgxPool(pool *pgxpool.Pool) *PgxPool {
	return &PgxPool{Pool: pool, driverName: "postgres", Mapper: mapper()}
}

// DriverName returns the driverName passed to the Open function for this DB.
func (db *PgxPool) DriverName() string {
	return db.driverName
}

// MapperFunc sets a new mapper for this db using the default sqlx struct tag
// and the provided mapper function.
func (db *PgxPool) MapperFunc(mf func(string) string) {
	db.Mapper = reflectx.NewMapperFunc("db", mf)
}

// Rebind transforms a query from QUESTION to the DB driver's bindvar type.
func (db *PgxPool) Rebind(query string) string {
	return Rebind(BindType(db.driverName), query)
}

// Unsafe returns a version of DB which will silently succeed to scan when
// columns in the SQL result have no fields in the destination struct.
// sqlx.Stmt and sqlx.Tx which are created from this DB will inherit its
// safety behavior.
func (db *PgxPool) Unsafe() *PgxPool {
	return &PgxPool{Pool: db.Pool, driverName: db.driverName, unsafe: true, Mapper: db.Mapper}
}

// BindNamed binds a query using the DB driver's bindvar type.
func (db *PgxPool) BindNamed(query string, arg interface{}) (string, []interface{}, error) {
	return bindNamedMapper(BindType(db.driverName), query, arg, db.Mapper)
}

// NamedQuery using this DB.
// Any named placeholder parameters are replaced with fields from arg.
func (db *PgxPool) NamedQuery(ctx context.Context, query string, arg interface{}) (*PgxRows, error) {
	return NamedPgxQuery(ctx, db, query, arg)
}

// NamedExec using this DB.
// Any named placeholder parameters are replaced with fields from arg.
func (db *PgxPool) NamedExec(ctx context.Context, query string, arg interface{}) (pgconn.CommandTag, error) {
	return NamedPgxExec(ctx, db, query, arg)
}

// Select using this DB.
// Any placeholder parameters are replaced with supplied args.
func (db *PgxPool) Select(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return SelectPgx(ctx, db, dest, query, args...)
}

// Get using this DB.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func (db *PgxPool) Get(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return GetPgx(ctx, db, dest, query, args...)
}

// MustBegin starts a transaction, and panics on error.  Returns an *sqlx.Tx instead
// of an *sql.Tx.
func (db *PgxPool) MustBegin(ctx context.Context) *PgxTx {
	tx, err := db.Beginx(ctx)
	if err != nil {
		panic(err)
	}
	return tx
}

// Beginx begins a transaction and returns an *sqlx.Tx instead of an *sql.Tx.
func (db *PgxPool) Beginx(ctx context.Context) (*PgxTx, error) {
	tx, err := db.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	return &PgxTx{Tx: tx, driverName: db.driverName, unsafe: db.unsafe, Mapper: db.Mapper}, err
}

// Queryx queries the database and returns an *sqlx.Rows.
// Any placeholder parameters are replaced with supplied args.
func (db *PgxPool) Queryx(ctx context.Context, query string, args ...interface{}) (*PgxRows, error) {
	r, err := db.Pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return &PgxRows{Rows: r, unsafe: db.unsafe, Mapper: db.Mapper}, err
}

// QueryRowx queries the database and returns an *sqlx.Row.
// Any placeholder parameters are replaced with supplied args.
func (db *PgxPool) QueryRowx(ctx context.Context, query string, args ...interface{}) *PgxRow {
	rows, err := db.Pool.Query(ctx, query, args...)
	return &PgxRow{rows: rows, err: err, unsafe: db.unsafe, Mapper: db.Mapper}
}

// PgxTx is an sqlx wrapper around sql.Tx with extra functionality
type PgxTx struct {
	pgx.Tx
	driverName string
	unsafe     bool
	Mapper     *reflectx.Mapper
}

// DriverName returns the driverName used by the DB which began this transaction.
func (tx *PgxTx) DriverName() string {
	return tx.driverName
}

// Rebind a query within a transaction's bindvar type.
func (tx *PgxTx) Rebind(query string) string {
	return Rebind(BindType(tx.driverName), query)
}

// Unsafe returns a version of Tx which will silently succeed to scan when
// columns in the SQL result have no fields in the destination struct.
func (tx *PgxTx) Unsafe() *PgxTx {
	return &PgxTx{Tx: tx.Tx, driverName: tx.driverName, unsafe: true, Mapper: tx.Mapper}
}

// BindNamed binds a query within a transaction's bindvar type.
func (tx *PgxTx) BindNamed(query string, arg interface{}) (string, []interface{}, error) {
	return bindNamedMapper(BindType(tx.driverName), query, arg, tx.Mapper)
}

// NamedQuery within a transaction.
// Any named placeholder parameters are replaced with fields from arg.
func (tx *PgxTx) NamedQuery(ctx context.Context, query string, arg interface{}) (*PgxRows, error) {
	return NamedPgxQuery(ctx, tx, query, arg)
}

// NamedExec a named query within a transaction.
// Any named placeholder parameters are replaced with fields from arg.
func (tx *PgxTx) NamedExec(ctx context.Context, query string, arg interface{}) (pgconn.CommandTag, error) {
	return NamedPgxExec(ctx, tx, query, arg)
}

// Select within a transaction.
// Any placeholder parameters are replaced with supplied args.
func (tx *PgxTx) Select(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return SelectPgx(ctx, tx, dest, query, args...)
}

// Queryx within a transaction.
// Any placeholder parameters are replaced with supplied args.
func (tx *PgxTx) Queryx(ctx context.Context, query string, args ...interface{}) (*PgxRows, error) {
	r, err := tx.Tx.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return &PgxRows{Rows: r, unsafe: tx.unsafe, Mapper: tx.Mapper}, err
}

// QueryRowx within a transaction.
// Any placeholder parameters are replaced with supplied args.
func (tx *PgxTx) QueryRowx(ctx context.Context, query string, args ...interface{}) *PgxRow {
	rows, err := tx.Tx.Query(ctx, query, args...)
	return &PgxRow{rows: rows, err: err, unsafe: tx.unsafe, Mapper: tx.Mapper}
}

// Get within a transaction.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func (tx *PgxTx) Get(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return GetPgx(ctx, tx, dest, query, args...)
}

// PgxRows is a wrapper around pgx.Rows which caches costly reflect operations
// during a looped StructScan
type PgxRows struct {
	pgx.Rows
	unsafe bool
	Mapper *reflectx.Mapper
	// these fields cache memory use for a rows during iteration w/ structScan
	started bool
	fields  [][]int
	values  []interface{}
}

// Close calls the underlying close method
func (r *PgxRows) Close() error {
	r.Rows.Close()
	return nil
}

// SliceScan using this Rows.
func (r *PgxRows) SliceScan() ([]interface{}, error) {
	return SliceScan(r)
}

// MapScan using this Rows.
func (r *PgxRows) MapScan(dest map[string]interface{}) error {
	return MapScan(r, dest)
}

// Columns returns the names of all columns
func (r *PgxRows) Columns() ([]string, error) {
	// From github.com/jackc/pgx/stdlib/sql.go method Columns
	fieldDescriptions := r.FieldDescriptions()
	names := make([]string, 0, len(fieldDescriptions))
	for _, fd := range fieldDescriptions {
		names = append(names, string(fd.Name))
	}
	return names, nil
}

// StructScan is like sql.Rows.Scan, but scans a single Row into a single Struct.
// Use this and iterate over Rows manually when the memory load of Select() might be
// prohibitive.  *Rows.StructScan caches the reflect work of matching up column
// positions to fields to avoid that overhead per scan, which means it is not safe
// to run StructScan on the same Rows instance with different struct types.
func (r *PgxRows) StructScan(dest interface{}) error {
	v := reflect.ValueOf(dest)

	if v.Kind() != reflect.Ptr {
		return errors.New("must pass a pointer, not a value, to StructScan destination")
	}

	v = v.Elem()

	if !r.started {
		columns, err := r.Columns()
		if err != nil {
			return err
		}
		m := r.Mapper

		r.fields = m.TraversalsByName(v.Type(), columns)
		// if we are not unsafe and are missing fields, return an error
		if f, err := missingFields(r.fields); err != nil && !r.unsafe {
			return fmt.Errorf("missing destination name %s in %T", columns[f], dest)
		}
		r.values = make([]interface{}, len(columns))
		r.started = true
	}

	err := fieldsByTraversal(v, r.fields, r.values, true)
	if err != nil {
		return err
	}
	// scan into the struct field pointers and append to our results
	err = r.Scan(r.values...)
	if err != nil {
		return err
	}
	return r.Err()
}

// SliceScan using this Rows.
func (r *PgxRow) SliceScan() ([]interface{}, error) {
	return SliceScan(r)
}

// MapScan using this Rows.
func (r *PgxRow) MapScan(dest map[string]interface{}) error {
	return MapScan(r, dest)
}

func (r *PgxRow) scanAny(dest interface{}, structOnly bool) error {
	if r.err != nil {
		return r.err
	}
	if r.rows == nil {
		r.err = sql.ErrNoRows
		return r.err
	}
	defer r.rows.Close()

	v := reflect.ValueOf(dest)
	if v.Kind() != reflect.Ptr {
		return errors.New("must pass a pointer, not a value, to StructScan destination")
	}
	if v.IsNil() {
		return errors.New("nil pointer passed to StructScan destination")
	}

	base := reflectx.Deref(v.Type())
	scannable := isScannable(base)

	if structOnly && scannable {
		return structOnlyError(base)
	}

	columns, err := r.Columns()
	if err != nil {
		return err
	}

	if scannable && len(columns) > 1 {
		return fmt.Errorf("scannable dest type %s with >1 columns (%d) in result", base.Kind(), len(columns))
	}

	if scannable {
		return r.Scan(dest)
	}

	m := r.Mapper

	fields := m.TraversalsByName(v.Type(), columns)
	// if we are not unsafe and are missing fields, return an error
	if f, err := missingFields(fields); err != nil && !r.unsafe {
		return fmt.Errorf("missing destination name %s in %T", columns[f], dest)
	}
	values := make([]interface{}, len(columns))

	err = fieldsByTraversal(v, fields, values, true)
	if err != nil {
		return err
	}
	// scan into the struct field pointers and append to our results
	return r.Scan(values...)
}

// StructScan a single Row into dest.
func (r *PgxRow) StructScan(dest interface{}) error {
	return r.scanAny(dest, true)
}

// SelectPgx executes a query using the provided Queryer, and StructScans each row
// into dest, which must be a slice.  If the slice elements are scannable, then
// the result set must have only one column.  Otherwise, StructScan is used.
// The *sql.Rows are closed automatically.
// Any placeholder parameters are replaced with supplied args.
func SelectPgx(ctx context.Context, q PgxQueryer, dest interface{}, query string, args ...interface{}) error {
	rows, err := q.Queryx(ctx, query, args...)
	if err != nil {
		return err
	}
	// if something happens here, we want to make sure the rows are Closed
	defer rows.Close()
	return scanAll(rows, dest, false)
}

// GetPgx does a QueryRow using the provided Queryer, and scans the resulting row
// to dest.  If dest is scannable, the result must only have one column.  Otherwise,
// StructScan is used.  Get will return sql.ErrNoRows like row.Scan would.
// Any placeholder parameters are replaced with supplied args.
// An error is returned if the result set is empty.
func GetPgx(ctx context.Context, q PgxQueryer, dest interface{}, query string, args ...interface{}) error {
	r := q.QueryRowx(ctx, query, args...)
	return r.scanAny(dest, false)
}

// NamedPgxQuery binds a named query and then runs Query on the result using the
// provided Ext (sqlx.Tx, sqlx.Db).  It works with both structs and with
// map[string]interface{} types.
func NamedPgxQuery(ctx context.Context, e PgxExt, query string, arg interface{}) (*PgxRows, error) {
	q, args, err := bindNamedMapper(BindType(e.DriverName()), query, arg, mapperFor(e))
	if err != nil {
		return nil, err
	}
	return e.Queryx(ctx, q, args...)
}

// NamedPgxExec uses BindStruct to get a query executable by the driver and
// then runs Exec on the result.  Returns an error from the binding
// or the query execution itself.
func NamedPgxExec(ctx context.Context, e PgxExt, query string, arg interface{}) (pgconn.CommandTag, error) {
	q, args, err := bindNamedMapper(BindType(e.DriverName()), query, arg, mapperFor(e))
	if err != nil {
		return nil, err
	}
	return e.Exec(ctx, q, args...)
}
